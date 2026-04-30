"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117

从本地 parquet 文件中读取文本数据；
按分布式训练规则（DDP 分片）迭代数据；
将文本 token 化并按「BOS 对齐 + 最佳拟合填充」策略打包成训练批次；
支持断点续训（记录数据迭代位置）；
输出模型可直接使用的inputs/targets张量。
"""

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size, data_dir=None):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    
    概念	含义
Parquet 文件	数据集的存储格式，dataset.py 下载的是多个 parquet 分片文件（比如 240 个）；
Row Group（RG）	Parquet 文件内部的「数据块」，每个 parquet 文件被分成多个 RG，是读取的最小单位；
DDP 分片规则	假设总共有 ddp_world_size 个进程（比如 8 卡 = 8 进程），进程ddp_rank（0~7）只读取 rg_idx = rank, rank+world_size, rank+2*world_size... 的 RG；
断点续训状态	resume_state_dict 包含 pq_idx（当前读到第几个 parquet 文件）、rg_idx（当前读到该文件的第几个 RG）、epoch（当前轮次）；
    
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" and data_dir is None # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(data_dir=data_dir, warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    data_dir=None,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    将文本批次 token 化，并按「BOS 对齐 + 最佳拟合填充」打包成训练批次
    BOS 对齐：每个训练行以 BOS token 开头，所有 token 都能 attention 到 BOS，保证上下文完整；减少模型学习的「混乱 token」，提升训练效果
    最佳拟合填充：1. 优先选最大的能完整放入剩余空间的文档；2. 无文档适配时，裁剪最短文档填充剩余空间；00% 显存利用率（无 padding），最小化裁剪损失
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size, data_dir=data_dir)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    #预分配固定大小的缓冲区，避免训练时频繁创建 / 销毁张量，同时通过pin_memory和「单次 HtoD 传输」最大化数据传输效率
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    #1. row_buffer：临时存储B个序列（每个长度T+1），用于构建每一批的token序列
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    # 2. cpu_buffer：CPU端的固定缓冲区，分两部分：前B*T是inputs，后B*T是targets（pin_memory=True加速CPU→GPU传输）
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    # 3. gpu_buffer：GPU端的固定缓冲区，和cpu_buffer结构一致，最终返回给训练代码
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents 保证缓冲区有足够文档（另外可以对比首次适配或者下次适配等内存管理算法学习）
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely第一步：找「最大的能完整放入剩余空间」的文
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets


def mixed_tokenizing_dataloader(
    tokenizer, B, T, split,
    data_dirs_with_weights,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    A multi-source dataloader wrapper that multiplexes multiple data_dirs
    using a provided weighting (e.g., [2, 2, 1]).
    Handles resume tracking perfectly across all streams by returning
    a composite nested state dict.
    
    data_dirs_with_weights: List of tuples [(data_dir, weight), ...]
    """
    # Parse ratios to build the sampling schedule sequence
    # E.g. [("english", 2), ("chinese", 2), ("finance", 1)] -> [0, 0, 1, 1, 2]
    schedule = []
    for stream_idx, (path, weight) in enumerate(data_dirs_with_weights):
        schedule.extend([stream_idx] * weight)
    
    # Resume configuration
    if resume_state_dict is None:
        resume_states = [None] * len(data_dirs_with_weights)
        cycle_idx = 0
    else:
        resume_states = resume_state_dict.get("states", [None] * len(data_dirs_with_weights))
        cycle_idx = resume_state_dict.get("cycle_idx", 0)

    # Initialize all internal dataloaders
    loaders = []
    for stream_idx, (path, _) in enumerate(data_dirs_with_weights):
        loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer=tokenizer, B=B, T=T, split=split,
            data_dir=path,
            tokenizer_threads=tokenizer_threads,
            tokenizer_batch_size=tokenizer_batch_size,
            device=device,
            resume_state_dict=resume_states[stream_idx],
            buffer_size=buffer_size
        )
        loaders.append(loader)

    # Maintain an internal representation of the states to yield out
    # Instead of fetching state_dict on every iteration (which requires iterating that generator),
    # we update the corresponding sub-state when we sample from it.
    current_states = [{"pq_idx": 0, "rg_idx": 0, "epoch": 1} for _ in loaders]
    
    # We need to initialize the states if we are resuming (but starting from right stream offsets)
    if resume_state_dict is not None:
        current_states = resume_states.copy()

    while True:
        # Pick the appropriate dataloader based on the current schedule step
        stream_idx = schedule[cycle_idx]
        
        # Sample one batch from the selected dataloader
        inputs, targets, state = next(loaders[stream_idx])
        
        # Update the internal state for the selected stream
        current_states[stream_idx] = state
        
        # Advance the schedule
        next_cycle_idx = (cycle_idx + 1) % len(schedule)
        
        # Yield the nested state dictionary
        composite_state = {
            "states": current_states.copy(),
            "cycle_idx": next_cycle_idx,
            # We provide dummy top-level keys for backward compatibility with 
            # logging code that defaults to assuming a single dataloader.
            # In base_train.py we will update the print statement, but having these helps.
            "pq_idx": state["pq_idx"],
            "rg_idx": state["rg_idx"],
            "epoch": state["epoch"],
            "source_idx": stream_idx
        }
        
        yield inputs, targets, composite_state
        
        cycle_idx = next_cycle_idx
