import platform
import psutil
import GPUtil

# OS Information
print(f"Operating System: {platform.system()} {platform.release()} ({platform.version()})")
print(f"OS Architecture: {platform.architecture()[0]}")
print(f"Machine Type: {platform.machine()}")

# CPU Information
# a = platform.
cpu_info = platform.processor() or "CPU information not available"
print(f"CPU: {cpu_info}")
print(f"Physical cores: {psutil.cpu_count(logical=False)}")
print(f"Total cores: {psutil.cpu_count(logical=True)}")
print(f"CPU Frequency: {psutil.cpu_freq().current:.2f} MHz")

# GPU Information
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU: {gpu.name}")
    print(f"GPU Driver: {gpu.driver}")
    print(f"GPU Memory: {gpu.memoryTotal}MB")
