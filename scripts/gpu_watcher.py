import subprocess
import time
import sys
import os

# --- Configuration ---
# 显卡计算占用率阈值 (百分比).
IDLE_THRESHOLD = 10

# 显卡显存占用阈值 (MB). 
# 设定为 1024 MB，如果占用低于 1GB 则认为是可用空卡.
IDLE_MEM_THRESHOLD_MB = 1024 

# 检查间隔 (秒). 测试时可以改小一点，比如 60 秒.
CHECK_INTERVAL_SECONDS = 3000

# 需要请求的显卡数量
NUM_GPUS_TO_REQUEST = 2
# ---

def get_gpu_stats():
    """
    使用 nvidia-smi 获取显卡状态并解析。
    包含 index, utilization, memory_used 和 memory_total。
    """
    try:
        # 修改了命令，增加了 memory.used 和 memory.total 的查询
        command = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        
        output = result.stdout.strip()
        if not output:
            print("No GPUs found by nvidia-smi.")
            return None

        gpus = []
        lines = output.split('\n')
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 4:
                gpus.append({
                    "index": int(parts[0]),
                    "utilization": int(parts[1]),
                    "memory_used": int(parts[2]),       # 新增：已用显存
                    "memory_total": int(parts[3])       # 新增：总显存
                })
        return gpus

    except FileNotFoundError:
        print("Error: 'nvidia-smi' command not found. Is the NVIDIA driver installed?")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi: {e.stderr}")
        return None

def run_training_script(script_path, gpu_indices):
    """
    在指定的 GPU 上执行训练脚本
    """
    gpu_indices_str = ",".join(map(str, gpu_indices))
    print(f"Executing training script '{script_path}' on GPUs: {gpu_indices_str}...")

    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_indices_str
    
    # 增加 NCCL 相关的环境变量，防止因为单卡 OOM 导致全员死锁不退出
    env["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env["TORCH_NCCL_BLOCKING_WAIT"] = "1" 
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    try:
        process = subprocess.Popen(
            ['bash', script_path], 
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1, # 行缓冲
            encoding='utf-8',
            errors='replace'
        )

        # 实时打印输出流的更优写法，防止缓冲区死锁
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.stdout.close()
        return_code = process.wait()

        if return_code == 0:
            print(f"\nTraining script '{script_path}' finished successfully.")
        else:
            print(f"\nTraining script '{script_path}' finished with error code {return_code}.")

    except FileNotFoundError:
        print("Error: The shell 'bash' was not found.")
    except Exception as e:
        print(f"An error occurred while running the script: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python gpu_watcher.py <path_to_your_training_script.sh>")
        sys.exit(1)

    training_script = sys.argv[1]

    if not os.path.exists(training_script):
        print(f"Error: The specified script file does not exist: {training_script}")
        sys.exit(1)
        
    print("--- GPU Training Watcher ---")
    print(f"Training script: {training_script}")
    print(f"Idle thresholds: Compute < {IDLE_THRESHOLD}%, VRAM < {IDLE_MEM_THRESHOLD_MB} MB")
    print(f"Check interval: {CHECK_INTERVAL_SECONDS} seconds")
    print(f"Number of GPUs to request: {NUM_GPUS_TO_REQUEST}")
    print("----------------------------")

    all_gpus = get_gpu_stats()
    if all_gpus is None:
        sys.exit(1)

    if len(all_gpus) < NUM_GPUS_TO_REQUEST:
        print(f"Error: Found only {len(all_gpus)} GPU(s), but {NUM_GPUS_TO_REQUEST} are required.")
        sys.exit(1)
    
    print(f"Found {len(all_gpus)} total GPUs. Starting monitoring...")

    while True:
        gpus = get_gpu_stats()
        
        if gpus is None:
            print(f"Could not retrieve GPU stats. Retrying in {CHECK_INTERVAL_SECONDS} seconds...")
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        # 过滤条件修改：同时检查 GPU 使用率和显存使用量
        idle_gpus = [
            gpu for gpu in gpus 
            if gpu['utilization'] < IDLE_THRESHOLD and gpu['memory_used'] < IDLE_MEM_THRESHOLD_MB
        ]

        if len(idle_gpus) >= NUM_GPUS_TO_REQUEST:
            # 优先使用显存占用最小的显卡
            idle_gpus.sort(key=lambda x: x['memory_used'])
            selected_gpus = idle_gpus[:NUM_GPUS_TO_REQUEST]
            selected_indices = [gpu['index'] for gpu in selected_gpus]
            
            # 打印被选中的卡的状态
            util_str = ", ".join([f"GPU {g['index']} (Util: {g['utilization']}%, VRAM: {g['memory_used']}MB)" for g in selected_gpus])
            print(f"\nFound {len(selected_gpus)} available GPUs: {util_str}.")
            
            run_training_script(training_script, selected_indices)
            break 
        else:
            print(f"Found {len(idle_gpus)} idle GPUs, but {NUM_GPUS_TO_REQUEST} are required. Waiting... (Press Ctrl+C to exit)")
            try:
                time.sleep(CHECK_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                print("Exiting watcher.")
                break

if __name__ == "__main__":
    main()
