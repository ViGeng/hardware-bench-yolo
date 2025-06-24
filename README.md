# Hardware Benchmarks for YOLO

Though manufacturers provide performance specifications for their hardware, these numbers are often not representative of real-world performance or are not always intuitive to interpret. This toy project aims to provide a collective benchmark of various hardware configurations for commonly used models such as YOLO series.

If you happen to have access to any hardware or performance metrics, please feel free to contribute by submitting a pull request. The goal is to create a comprehensive benchmark that can help others make informed decisions when selecting hardware for their machine learning tasks or real-world applications.

## Benchmark Specifications

To ensure the accuracy and reliability of the benchmarks, we will be using same datasets and models across all hardware configurations. 

### Dataset

**Videos**: [sample-videos](https://github.com/intel-iot-devkit/sample-videos)

### Models

YOLO Series by [Ultralytics](https://github.com/ultralytics/ultralytics)

### Metrics

* Throughput
* Latency (different stages)
* Resource usage (CPU, GPU)

### Potential Future Works

- More typical workload (Detection, Classification, or CV tasks)
- More hardwares
- Container based deliverables for convenience reproduction
