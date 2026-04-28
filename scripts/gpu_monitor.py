#!/usr/bin/env python3
"""
GPU 显存占用器
功能：监控所有 GPU，当发现某张卡的显存使用率低于阈值时，
      自动在该卡上分配指定比例（默认 83%）的显存，并保持占用直到用户中断。
依赖：torch, nvidia-ml-py (或 pynvml)
"""

import time
import argparse
import sys

# 尝试导入 nvidia-ml-py（推荐），回退到 pynvml
try:
    import nvidia_smi
    nvidia_smi.nvmlInit()
    _nvml_handle = nvidia_smi
except ImportError:
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_handle = pynvml
        print("警告：pynvml 已弃用，建议安装 nvidia-ml-py (pip install nvidia-ml-py)", file=sys.stderr)
    except ImportError:
        raise ImportError("需要安装 nvidia-ml-py 或 pynvml")

import torch

def parse_args():
    parser = argparse.ArgumentParser(description="GPU 显存占用器")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="显存使用率阈值（0~1），低于此值视为空闲，默认 0.1 (10%%)")
    parser.add_argument("--target_ratio", type=float, default=0.83,
                        help="目标占用比例（0~1），默认 0.83 (83%%)")
    parser.add_argument("--interval", type=int, default=100,
                        help="检查间隔（秒），默认 180 (3 分钟)")
    parser.add_argument("--gpu_count", type=int, default=8,
                        help="期望的 GPU 数量，用于校验，默认 8")
    parser.add_argument("--no_check_count", action="store_true",
                        help="跳过 GPU 数量校验")
    return parser.parse_args()

def get_device_count():
    """返回可用的 GPU 数量"""
    return _nvml_handle.nvmlDeviceGetCount()

def get_memory_usage(device_index):
    """返回指定 GPU 的显存使用率 (0~1)"""
    handle = _nvml_handle.nvmlDeviceGetHandleByIndex(device_index)
    info = _nvml_handle.nvmlDeviceGetMemoryInfo(handle)
    return info.used / info.total

def get_total_memory(device_index):
    """返回指定 GPU 的总显存（字节）"""
    handle = _nvml_handle.nvmlDeviceGetHandleByIndex(device_index)
    info = _nvml_handle.nvmlDeviceGetMemoryInfo(handle)
    return info.total

def allocate_memory(device_index, target_ratio):
    """
    在指定 GPU 上分配显存，占用比例为 target_ratio（相对于总显存）
    返回分配的张量，若失败则返回 None
    """
    total_bytes = get_total_memory(device_index)
    target_bytes = int(total_bytes * target_ratio)
    # float32 每个元素 4 字节
    num_elements = target_bytes // 4
    try:
        tensor = torch.zeros(num_elements, dtype=torch.float32, device=f'cuda:{device_index}')
        print(f"成功在 GPU {device_index} 上分配了 {target_bytes / (1024**3):.2f} GB 显存 (占用比例 {target_ratio*100:.1f}%)")
        return tensor
    except RuntimeError as e:
        print(f"分配失败 (GPU {device_index}): {e}", file=sys.stderr)
        return None

def main():
    args = parse_args()

    # 初始化 NVML
    # 已在导入时初始化，这里可再次确认
    if hasattr(_nvml_handle, 'nvmlInit'):
        _nvml_handle.nvmlInit()

    # 获取 GPU 数量并校验
    device_count = get_device_count()
    if not args.no_check_count:
        if device_count != args.gpu_count:
            print(f"错误：检测到 {device_count} 张 GPU，期望 {args.gpu_count} 张，请检查或使用 --no_check_count 跳过", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"检测到 {device_count} 张 GPU")

    print(f"监控开始：阈值 {args.threshold*100:.0f}%，目标占用 {args.target_ratio*100:.0f}%，检查间隔 {args.interval} 秒")
    occupied_tensor = None
    try:
        while True:
            time.sleep(args.interval)
            for i in range(device_count):
                usage = get_memory_usage(i)
                if usage < args.threshold:
                    print(f"发现空闲 GPU {i}，当前使用率 {usage*100:.1f}%")
                    # 尝试分配显存
                    occupied_tensor = allocate_memory(i, args.target_ratio)
                    if occupied_tensor is not None:
                        print(f"已占用 GPU {i}，按 Ctrl+C 退出释放显存")
                        # 保持程序运行
                        while True:
                            time.sleep(1)
                    else:
                        print(f"占用 GPU {i} 失败，继续监控")
                        break  # 跳出内循环，继续外循环
            else:
                # 未找到空闲 GPU，继续监控
                print(f"当前无空闲 GPU，{args.interval} 秒后重试")
                continue
            # 如果成功占用，上面的内层 while True 会一直循环，不会走到这里
            # 如果分配失败，则 break 出来后会继续外循环
            # 这里无需额外处理
    except KeyboardInterrupt:
        print("\n用户中断，释放显存并退出")
    finally:
        # 清理：删除张量，释放显存
        if occupied_tensor is not None:
            del occupied_tensor
            torch.cuda.empty_cache()
            print("显存已释放")
        # 关闭 NVML
        if hasattr(_nvml_handle, 'nvmlShutdown'):
            _nvml_handle.nvmlShutdown()

if __name__ == "__main__":
    main()