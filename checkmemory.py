import os
import psutil
import torch

def print_top_cpu_memory_processes(n=10):
    """Print the top n processes sorted by CPU memory usage (RSS)."""
    print("Top processes by CPU memory usage:")
    processes = []
    for p in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            mem = p.info['memory_info'].rss  # in bytes
            processes.append((p.info['pid'], p.info['name'], mem))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    # Sort descending by memory usage
    processes.sort(key=lambda x: x[2], reverse=True)
    for pid, name, mem in processes[:n]:
        print(f"PID: {pid:<6} Name: {name:<25} Memory: {mem / (1024 ** 2):>6.2f} MB")
    print("\n")

def print_current_process_memory():
    """Print the memory usage of the current process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Current process memory usage:")
    print(f"  RSS (Resident Set Size): {mem_info.rss / (1024 ** 2):.2f} MB")
    print(f"  VMS (Virtual Memory Size): {mem_info.vms / (1024 ** 2):.2f} MB")
    print("\n")

def print_gpu_memory_summary():
    """Print a summary of GPU memory usage using PyTorch (if CUDA is available)."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{i}")
            print(f"GPU {i} ({torch.cuda.get_device_name(device)}):")
            print(torch.cuda.memory_summary(device=device, abbreviated=True))
            print("-" * 80)
    else:
        print("No CUDA devices available.")

if __name__ == "__main__":
    print_current_process_memory()
    print_top_cpu_memory_processes(n=10)
    print_gpu_memory_summary()
