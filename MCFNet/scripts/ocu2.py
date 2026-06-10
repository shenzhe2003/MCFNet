
import torch
import time
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='LGMFNet_For_SIRST')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--memory', type=int, default=20000)
    args = parser.parse_args()
    return args
args = parse_args()

#  确保使用的是 GPU
if torch.cuda.is_available():
    device = torch.device("cuda:"+args.gpu)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU instead.")

# 定义要占用的显存大小（以MB为单位）
target_mem_size_mb = args.memory # 想占用的总显存大小：

# 转换为字节数
target_mem_size_bytes = target_mem_size_mb * 1024 * 1024

# 每次分配的张量大小（以MB为单位），避免一次分配太大
chunk_size_mb = 1000 # 每次分配1GB
chunk_size_bytes = chunk_size_mb * 1024 * 1024

# 计算每个张量需要的元素数量（float32，每个4字节）
num_elements_per_chunk = chunk_size_bytes // 4

# 分配多个小张量来逼近目标显存大小
tensors = []
allocated_mem = 0  # 记录分配的显存总量

try:
    while allocated_mem < target_mem_size_bytes:
        tensor = torch.empty(num_elements_per_chunk, dtype=torch.float32, device=device)
        tensors.append(tensor)
        allocated_mem += chunk_size_bytes
        print(f"Allocated {allocated_mem / (1024 * 1024)} MB of GPU memory.")
except RuntimeError as e:
    print(f"Failed to allocate more memory: {e}")

print(f"Final allocated memory: {allocated_mem / (1024 * 1024)} MB")

# 让程序保持运行，不释放显存
try:
    while True:
        time.sleep(10)  # 每10秒暂停，保持显存占用
except KeyboardInterrupt:
    print("Program interrupted and exiting.")
