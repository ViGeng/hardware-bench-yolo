# Hardware Benchmarks for YOLO

Though manufacturers provide performance specifications for their hardware, these numbers are often not representative of real-world performance or are not always intuitive to interpret. This toy project aims to provide a collective benchmark of various hardware configurations for commonly used models such as YOLO series.

If you happen to have access to any hardware or performance metrics, please feel free to contribute by submitting a pull request. The goal is to create a comprehensive benchmark that can help others make informed decisions when selecting hardware for their machine learning tasks or real-world applications.

## Benchmark Specifications

To ensure the accuracy and reliability of the benchmarks, we will be using same datasets and models across all hardware configurations. 

### Dataset

**Videos**: [sample-videos](https://github.com/intel-iot-devkit/sample-videos)
* other common detection datasets (coco, VoC, etc)
* live stream (pratical live scenarios, with diff resolutions, or conditions)

### Models

YOLO Series by [Ultralytics](https://github.com/ultralytics/ultralytics)

### Metrics

* Throughput
* Latency (different stages)
* Resource usage (CPU, GPU)

### Devices

* Powerful Nvidia GPUs
* Intel CPUs
* Mobile SoCs
* Edge Devices (such as Raspberry Pis)

### Potential Future Works

- More typical workload (Detection, Classification, or CV tasks)
- More hardwares
- Container based deliverables for convenience reproduction






# #####################################################################


# benchmark_improved #

YOLOv8 基准测试工具 - 使用说明
文件说明
benchmark.py - 原版基准测试程序
benchmark_improved.py - 改进版基准测试程序（推荐使用）
requirements.txt - 依赖包列表
改进功能
命令行参数支持 - 不用修改代码就能改变测试参数
多种输出格式 - 支持 txt、json、csv 三种格式
更好的错误处理 - 自动检测文件和设备
详细的进度显示 - 可以看到实时处理进度
基本使用方法
1. 最简单的使用（测试摄像头）
bash
python benchmark_improved.py
2. 测试视频文件
bash
python benchmark_improved.py --source "D:/samplevideos/bolt-detection.mp4"
3. 使用不同模型
bash
python benchmark_improved.py --model yolov8s.pt --source test.mp4
4. 指定GPU设备
bash
python benchmark_improved.py --device cuda:0 --source test.mp4
5. 限制测试帧数（快速测试）
bash
python benchmark_improved.py --source test.mp4 --max-frames 100
6. 输出多种格式
bash
python benchmark_improved.py --source test.mp4 --output-format txt json csv
7. 显示详细信息
bash
python benchmark_improved.py --source test.mp4 --verbose
完整参数列表
参数	简写	默认值	说明
--model	-m	yolov8n.pt	模型路径或名称
--source	-s	0	视频源（文件路径或摄像头号）
--device	-d	auto	运行设备（cpu/cuda:0/auto）
--batch-size	-b	1	批次大小
--max-frames	-f	无限制	最大处理帧数
--imgsz	无	模型默认	输入图像尺寸
--output-format	-o	txt	输出格式（txt/json/csv）
--verbose	-v	关闭	显示详细信息
输出文件说明
程序会在当前目录生成以下文件：

主机名_时间戳.txt - 详细的文本报告
主机名_时间戳.json - JSON格式数据（便于程序处理）
主机名_时间戳.csv - CSV格式数据（可用Excel打开）
常用测试场景
快速性能测试
bash
python benchmark_improved.py --source 0 --max-frames 50 --verbose
完整视频文件测试
bash
python benchmark_improved.py --source video.mp4 --output-format txt csv
GPU性能测试
bash
python benchmark_improved.py --device cuda:0 --batch-size 4 --source video.mp4
不同模型对比测试
bash
# 测试小模型
python benchmark_improved.py --model yolov8n.pt --source test.mp4 --output-format json

# 测试中等模型
python benchmark_improved.py --model yolov8s.pt --source test.mp4 --output-format json

# 测试大模型
python benchmark_improved.py --model yolov8m.pt --source test.mp4 --output-format json
故障排除
模型下载问题
第一次使用会自动下载模型，如果网络有问题：

bash
# 手动下载模型到当前目录
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
CUDA相关错误
如果遇到GPU错误：

bash
# 强制使用CPU
python benchmark_improved.py --device cpu --source test.mp4
内存不足
如果内存不够：

bash
# 减小批次大小
python benchmark_improved.py --batch-size 1 --source test.mp4
与原版对比
功能	原版	改进版
修改参数	需要改代码	命令行参数
输出格式	只有txt	txt/json/csv
错误处理	基本	完善
进度显示	无	有
帧数限制	无	有
设备选择	固定	可选择
