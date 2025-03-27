from ultralytics import YOLO
import numpy as np
import time
import psutil
import torch
import threading
import socket
from collections import deque
import pynvml  # For GPU utilization monitoring

# Load a pretrained model
model = YOLO("yolo11m.pt")

# Define path to video file
source = "./sample-videos"

# Initialize metrics collection
preprocess_times = []
inference_times = []
postprocess_times = []
total_frames = 0
start_time = time.time()

# Initialize resource monitoring
cpu_percentages = deque(maxlen=1000)
memory_usages = deque(maxlen=1000)
gpu_mem_usages = deque(maxlen=1000) if torch.cuda.is_available() else None
gpu_util_usages = deque(maxlen=1000) if torch.cuda.is_available() else None

# Flag to control resource monitoring thread
monitoring = True

# Resource monitoring function
def monitor_resources():
    # Initialize NVML for GPU monitoring if available
    if torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
        except Exception as e:
            print(f"Warning: Could not initialize NVML: {e}")
            handle = None
    else:
        handle = None
        
    # Get current process for more accurate CPU monitoring
    current_process = psutil.Process()
    
    while monitoring:
        # CPU usage (for this process only)
        cpu_percentages.append(current_process.cpu_percent(interval=0.1))
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usages.append(memory.percent)
        
        # GPU usage if available
        if torch.cuda.is_available():
            # Get current GPU memory usage
            gpu_mem_alloc = torch.cuda.memory_allocated(0)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory
            gpu_mem_percent = (gpu_mem_alloc / gpu_mem_total) * 100
            gpu_mem_usages.append(gpu_mem_percent)
            
            # Get GPU utilization using NVML
            if handle is not None:
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_usages.append(utilization.gpu)  # GPU utilization percentage
                except Exception as e:
                    # If there's an error, append the last value or 0
                    last_value = gpu_util_usages[-1] if gpu_util_usages else 0
                    gpu_util_usages.append(last_value)
            
        time.sleep(0.1)  # Sample every 100ms
        
    # Cleanup NVML
    if torch.cuda.is_available() and handle is not None:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

# Start resource monitoring in a separate thread
monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.daemon = True
monitor_thread.start()

# Run inference on the source
results = model(source, stream=True, batch=4)  # generator of Results objects

# Process results generator
for result in results:
    # Collect metrics for each frame
    preprocess_times.append(result.speed["preprocess"])
    inference_times.append(result.speed["inference"])
    postprocess_times.append(result.speed["postprocess"])
    total_frames += 1
    
    # Print individual frame metrics (optional)
    # print(f"Frame {total_frames}: {result.speed}")

# Stop resource monitoring
monitoring = False
if monitor_thread.is_alive():
    monitor_thread.join(timeout=1.0)

# Calculate elapsed time and throughput
end_time = time.time()
total_time = end_time - start_time
throughput = total_frames / total_time

# Get hostname for the output file
hostname = socket.gethostname()
output_filename = f"{hostname}.txt"

# Open the output file
with open(output_filename, 'w') as f:
    # Write summary statistics to file
    f.write("\n===== BENCHMARK SUMMARY =====\n")
    f.write(f"Total frames processed: {total_frames}\n")
    f.write(f"Total time elapsed: {total_time:.2f} seconds\n")
    f.write(f"Throughput: {throughput:.2f} frames per second\n")
    
    # Write detailed metrics to file
    f.write("\n===== DETAILED METRICS =====\n")
    f.write("Preprocess time (ms):\n")
    f.write(f"  Min: {np.min(preprocess_times):.2f}\n")
    f.write(f"  Max: {np.max(preprocess_times):.2f}\n")
    f.write(f"  Avg: {np.mean(preprocess_times):.2f}\n")
    
    f.write("\nInference time (ms):\n")
    f.write(f"  Min: {np.min(inference_times):.2f}\n")
    f.write(f"  Max: {np.max(inference_times):.2f}\n")
    f.write(f"  Avg: {np.mean(inference_times):.2f}\n")
    
    f.write("\nPostprocess time (ms):\n")
    f.write(f"  Min: {np.min(postprocess_times):.2f}\n")
    f.write(f"  Max: {np.max(postprocess_times):.2f}\n")
    f.write(f"  Avg: {np.mean(postprocess_times):.2f}\n")
    
    f.write("\nTotal processing time per frame (ms):\n")
    total_per_frame = [p + i + pp for p, i, pp in zip(preprocess_times, inference_times, postprocess_times)]
    f.write(f"  Min: {np.min(total_per_frame):.2f}\n")
    f.write(f"  Max: {np.max(total_per_frame):.2f}\n")
    f.write(f"  Avg: {np.mean(total_per_frame):.2f}\n")
    
    # Write resource utilization metrics to file
    f.write("\n===== RESOURCE UTILIZATION =====\n")
    f.write("CPU Usage (%):\n") 
    f.write(f"  Min: {np.min(cpu_percentages):.2f}\n")
    f.write(f"  Max: {np.max(cpu_percentages):.2f}\n")
    f.write(f"  Avg: {np.mean(cpu_percentages):.2f}\n")
    
    f.write("\nMemory Usage (%):\n") 
    f.write(f"  Min: {np.min(memory_usages):.2f}\n")
    f.write(f"  Max: {np.max(memory_usages):.2f}\n")
    f.write(f"  Avg: {np.mean(memory_usages):.2f}\n")
    
    if torch.cuda.is_available():
        f.write("\nGPU Memory Usage (%):\n") 
        f.write(f"  Min: {np.min(gpu_mem_usages):.2f}\n")
        f.write(f"  Max: {np.max(gpu_mem_usages):.2f}\n")
        f.write(f"  Avg: {np.mean(gpu_mem_usages):.2f}\n")
        
        f.write("\nGPU Utilization (%):\n") 
        f.write(f"  Min: {np.min(gpu_util_usages):.2f}\n")
        f.write(f"  Max: {np.max(gpu_util_usages):.2f}\n")
        f.write(f"  Avg: {np.mean(gpu_util_usages):.2f}\n")
        f.write(f"  Device: {torch.cuda.get_device_name(0)}\n")
    else:
        f.write("\nGPU: Not available\n")

# Print to console that results were saved
print(f"\nBenchmark results saved to {output_filename}")

# Calculate summary statistics for console output
print("\n===== BENCHMARK SUMMARY =====")
print(f"Total frames processed: {total_frames}")
print(f"Total time elapsed: {total_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} frames per second")

# Print detailed metrics
print("\n===== DETAILED METRICS =====")
print("Preprocess time (ms):")
print(f"  Min: {np.min(preprocess_times):.2f}")
print(f"  Max: {np.max(preprocess_times):.2f}")
print(f"  Avg: {np.mean(preprocess_times):.2f}")

print("\nInference time (ms):")
print(f"  Min: {np.min(inference_times):.2f}")
print(f"  Max: {np.max(inference_times):.2f}")
print(f"  Avg: {np.mean(inference_times):.2f}")

print("\nPostprocess time (ms):")
print(f"  Min: {np.min(postprocess_times):.2f}")
print(f"  Max: {np.max(postprocess_times):.2f}")
print(f"  Avg: {np.mean(postprocess_times):.2f}")

print("\nTotal processing time per frame (ms):")
total_per_frame = [p + i + pp for p, i, pp in zip(preprocess_times, inference_times, postprocess_times)]
print(f"  Min: {np.min(total_per_frame):.2f}")
print(f"  Max: {np.max(total_per_frame):.2f}")
print(f"  Avg: {np.mean(total_per_frame):.2f}")

# Print resource utilization metrics
print("\n===== RESOURCE UTILIZATION =====")
print("CPU Usage (%):") 
print(f"  Min: {np.min(cpu_percentages):.2f}")
print(f"  Max: {np.max(cpu_percentages):.2f}")
print(f"  Avg: {np.mean(cpu_percentages):.2f}")

print("\nMemory Usage (%):") 
print(f"  Min: {np.min(memory_usages):.2f}")
print(f"  Max: {np.max(memory_usages):.2f}")
print(f"  Avg: {np.mean(memory_usages):.2f}")

if torch.cuda.is_available():
    print("\nGPU Memory Usage (%):")
    print(f"  Min: {np.min(gpu_mem_usages):.2f}")
    print(f"  Max: {np.max(gpu_mem_usages):.2f}")
    print(f"  Avg: {np.mean(gpu_mem_usages):.2f}")
    
    print("\nGPU Utilization (%):")
    print(f"  Min: {np.min(gpu_util_usages):.2f}")
    print(f"  Max: {np.max(gpu_util_usages):.2f}")
    print(f"  Avg: {np.mean(gpu_util_usages):.2f}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
else:
    print("\nGPU: Not available")
