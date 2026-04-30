"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.

从一个数据集（通过 parquets_iter_batched 函数获取）中读取文本数据。
使用 RustBPETokenizer 库训练一个 BPE 分词器。
将训练好的分词器保存到指定目录。
执行一个快速的健全性检查，确保分词器可以正确编码和解码文本。
计算每个 token 对应的 UTF-8 字节数，并将结果保存，这对于后续计算 "bits per byte" 的评估指标非常重要。
记录训练参数和分词器相关的统计数据到报告系统。
"""

"""
日期：2026-04
作者：QH
描述：训练一个新的分词器，类似于 GPT-4 的 tokenizer。分词器将从通用英文数据集、通用中文数据集、中文金融数据集（通过 parquets_iter_batched 函数提供）中学习，并将训练好的分词器保存到磁盘。训练过程中会限制最大字符数和每个文档的最大字符数，以控制内存使用。训练完成后，还会计算每个 token 的字节数，并将这些信息保存以供后续评估使用。
路径说明：
- /mnt/disk/mxf/.cache/nanochat/base_data_climbmix : 通用英文数据集
- /home/mxf/projects/Qhhhhhhaaa/nanochat/finance-data-process/data/pre-data/Chinese_data : 通用中文数据集
- /home/mxf/projects/Qhhhhhhaaa/nanochat/finance-data-process/data/pre-data/Financial_data : 中文金融数据集
数据抽取说明：
英文通用抽取： ~500 MB（随便拿 2 个 Parquet 分片）
中文通用抽取： ~500 MB（拿 2-3 个 SkyPile/Wiki 分片）
中文金融抽取： ~500 MB（拿 1 个高浓度金融分片）
对应函数修改：
- parquets_iter_batched(split="train")：修改为从上述三个数据源中抽取文本数据，并进行适当的混合，以确保分词器能够学习到多样化的语言特征。可以通过参数控制每个数据源的抽取比例，例如 40% 英文通用、30% 中文通用、30% 中文金融。
- RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)：保持不变，继续使用 RustBPETokenizer 进行训练。
- 其他部分保持不变，继续计算 token 字节数并保存相关统计数据。
- 训练完成后，分词器将能够更好地处理混合中英文文本，特别是在金融领域的应用场景中表现更佳。减少后续CPT调整。
"""
import os
import time
import argparse
import random
import glob
import pyarrow.parquet as pq
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 10B)')#训练时读取的最大字符数，默认 20 亿字符，约合 20 GB 的文本数据。
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')# 每个文档的最大字符数，默认 10,000 字符。超过这个长度的文档会被截断。这是为了防止极长的文档占用过多内存。
parser.add_argument('--vocab-size', type=int, default=65000, help='Base vocab size (default: 65000, leaving space for special tokens)')# 词表大小，默认 65000。这是分词器将要学习的 token 数量。较大的词表可以更好地表示文本，并为后续追加特殊符预留空间。
args = parser.parse_args()  
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")
   
# -----------------------------------------------------------------------------
# Text iterator

def get_parquet_iterator(directory, num_files, text_col='text'):
    """获取指定目录下一定数量的 Parquet 文件的文本迭代器"""
    files = glob.glob(os.path.join(directory, "*.parquet"))
    # 安全采样：确保采样数量不会超过现有文件总数
    sample_size = min(num_files, len(files))
    if sample_size > 0:
        files = random.sample(files, sample_size)
    
    for f in files:
        try:
            parquet_file = pq.ParquetFile(f)
            for batch in parquet_file.iter_batches(columns=[text_col]):
                for text in batch[text_col].to_pylist():
                    if text:
                        yield text
        except Exception as e:
            print(f"Error reading {f}: {e}")

def text_iterator():
    """
    按照要求的比例混合三个数据源：
    - 英文通用 (40%, 权重 2)
    - 中文通用 (40%, 权重 2)
    - 中文金融 (20%, 权重 1)
    即 2:2:1 的配比
    """
    # 指定路径（根据你的注释）
    en_dir = "/mnt/disk/mxf/.cache/nanochat/base_data_climbmix"
    zh_dir = "/home/mxf/projects/Qhhhhhhaaa/nanochat/finance-data-process/data/pre-data/Chinese_data"
    fin_dir = "/home/mxf/projects/Qhhhhhhaaa/nanochat/finance-data-process/data/pre-data/Financial_data"

    # 初始化三个数据源的迭代器
    en_iter = get_parquet_iterator(en_dir, num_files=2)
    zh_iter = get_parquet_iterator(zh_dir, num_files=2)
    fin_iter = get_parquet_iterator(fin_dir, num_files=1)

    iters = [en_iter, zh_iter, fin_iter]
    # 调整为 2/2/1 的配比 (EN:ZH:FIN)
    weights = [2, 2, 1]
    population = [0, 1, 2]
    
    nchars = 0
    while True:
        if not population:
            print(f"All data sources exhausted. Total characters read: {nchars:,}")
            return

        # 按照权重随机选择一个数据源
        choice = random.choices(population, weights=weights, k=1)[0]
        try:
            doc_text = next(iters[choice])
            
            # 对超长文档进行截断
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
                
            nchars += len(doc_text)
            yield doc_text
            
            # 如果到达最大训练字符数，停止迭代
            if nchars > args.max_chars:
                print(f"Reached max characters: {nchars:,}")
                return
                
        except StopIteration:
            # 如果某个数据源耗尽了，从采样池和权重中剔除它
            idx = population.index(choice)
            weights.pop(idx)
            population.pop(idx)
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()#获取训练好的分词器的词表大小，应该等于我们在训练时指定的 args.vocab_size
special_set = set(tokenizer.get_special_tokens())#获取分词器的特殊 token 列表，并将其转换为一个集合（set）以便快速查找。特殊 token 通常包括像 <BOS>、<EOS>、<PAD> 这样的标记，这些标记在计算 token 字节数时会被特殊处理（通常不计入字节数）。
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] #通过解码 token_id 获取对应的 token 字符串表示。由于我们之前已经预先解码了所有 token_id 的字符串，这里直接从 token_strings 列表中获取对应的字符串。
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])
