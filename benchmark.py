from ultralytics import YOLO
import numpy as np
import time

# Load a pretrained model
model = YOLO("yolo11n.pt")

# Define path to video file
source = "./sample-videos"

# Initialize metrics collection
preprocess_times = []
inference_times = []
postprocess_times = []
total_frames = 0
start_time = time.time()

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects

# Process results generator
for result in results:
    # Collect metrics for each frame
    preprocess_times.append(result.speed["preprocess"])
    inference_times.append(result.speed["inference"])
    postprocess_times.append(result.speed["postprocess"])
    total_frames += 1
    
    # Print individual frame metrics (optional)
    # print(f"Frame {total_frames}: {result.speed}")

# Calculate elapsed time and throughput
end_time = time.time()
total_time = end_time - start_time
throughput = total_frames / total_time

# Calculate summary statistics
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
