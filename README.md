# Enhanced Deep Learning Benchmark Tool

一个模块化的深度学习模型基准测试工具，支持分类、检测和分割模型的性能评估。支持交互式界面和命令行参数两种使用方式。

## 项目结构

```
benchmark_project/
├── main.py              # 主程序入口
├── config.py            # 配置模块 - 模型和数据集配置
├── utils.py             # 工具模块 - 日志、依赖检查等
├── datasets.py          # 数据集模块 - 各种数据集加载
├── models.py            # 模型模块 - 模型加载和管理
├── rendering.py         # 渲染模块 - 结果可视化渲染
├── benchmarks.py        # 基准测试模块 - 测试逻辑
├── interactive.py       # 交互界面模块 - 用户交互
├── cli.py               # 命令行界面模块 - 命令行参数处理
├── monitoring.py        # 监控模块 - 资源监控和统计
├── output.py            # 输出模块 - 结果保存和图表生成
└── README.md            # 说明文档
```

## 功能特性

### 支持的模型类型
- **图像分类**: ResNet, EfficientNet, Vision Transformer, MobileNet等
- **目标检测**: YOLOv8系列, Faster R-CNN, FCOS等
- **语义分割**: DeepLabV3+, UNet, PSPNet, FPN等

### 支持的数据集
- **分类**: MNIST, CIFAR-10, ImageNet样本, 合成数据
- **检测**: COCO样本, KITTI, 预设测试图像, 合成数据
- **分割**: Cityscapes, 合成分割数据

### 核心功能
1. **交互式配置** - 支持设备、模型、数据集的灵活选择
2. **命令行支持** - 支持批量测试和自动化脚本
3. **详细性能分析** - 逐帧时间统计和FPS分析
4. **资源监控** - CPU、内存、GPU使用率实时监控
5. **结果可视化** - 自动生成性能图表和分析报告
6. **多格式输出** - CSV详细数据和PNG可视化图表
7. **完整日志系统** - 详细的执行日志记录

## 安装依赖

### 基础依赖
```bash
pip install torch torchvision numpy matplotlib seaborn psutil
```

### 可选依赖（推荐安装以获得完整功能）
```bash
# 图像分类模型
pip install timm

# 目标检测
pip install ultralytics

# 语义分割
pip install segmentation-models-pytorch

# GPU监控
pip install nvidia-ml-py3

# 图像处理
pip install Pillow opencv-python
```

## 使用方法

### 1. 交互式模式（默认）
```bash
python main.py
```

### 2. 命令行模式

#### 基本语法
```bash
python main.py --device <device> --model-type <type> --model <model> --dataset <dataset> --samples <num>
```

#### 常用命令示例

**查看帮助信息**
```bash
python main.py --help
```

**列出所有可用模型**
```bash
python main.py --list-models
```

**列出所有可用数据集**
```bash
python main.py --list-datasets
```

**使用CPU测试分类模型**
```bash
python main.py --device cpu --model-type classification --model resnet18 --dataset MNIST --samples 100
```

**使用GPU测试检测模型**
```bash
python main.py --device cuda:0 --model-type detection --model yolov8n --dataset Test-Images --samples 500
```

**使用GPU测试分割模型**
```bash
python main.py --device cuda:0 --model-type segmentation --model unet_resnet34 --dataset Synthetic-Segmentation --samples 200
```

**批量测试并保存到指定目录**
```bash
python main.py --device cuda:0 --model-type classification --model resnet50 --dataset CIFAR-10 --samples 1000 --output-dir ./results/resnet50_test
```

**静默模式测试（适合脚本）**
```bash
python main.py --device cuda:0 --model-type detection --model yolov8s --dataset Test-Images --samples 100 --quiet --no-plots
```

#### 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--device` | str | auto | 计算设备 (cpu, cuda:0, auto) |
| `--model-type` | str | - | 模型类型 (classification, detection, segmentation) |
| `--model` | str | - | 模型名称 |
| `--dataset` | str | - | 数据集名称 |
| `--samples` | int | 100 | 测试样本数量 (-1表示全部) |
| `--batch-size` | int | 1 | 批处理大小 |
| `--output-dir` | str | ./results | 输出目录 |
| `--no-plots` | flag | False | 不生成可视化图表 |
| `--quiet` | flag | False | 静默模式，减少输出 |
| `--list-models` | flag | False | 列出所有可用模型 |
| `--list-datasets` | flag | False | 列出所有可用数据集 |
| `--interactive` | flag | False | 强制使用交互模式 |

### 3. 批量脚本示例

**创建批量测试脚本**
```bash
#!/bin/bash
# 批量测试不同模型的性能

# 测试分类模型
python main.py --device cuda:0 --model-type classification --model resnet18 --dataset MNIST --samples 1000 --output-dir ./results/resnet18 --quiet
python main.py --device cuda:0 --model-type classification --model resnet50 --dataset MNIST --samples 1000 --output-dir ./results/resnet50 --quiet
python main.py --device cuda:0 --model-type classification --model efficientnet_b0 --dataset MNIST --samples 1000 --output-dir ./results/efficientnet_b0 --quiet

# 测试检测模型
python main.py --device cuda:0 --model-type detection --model yolov8n --dataset Test-Images --samples 500 --output-dir ./results/yolov8n --quiet
python main.py --device cuda:0 --model-type detection --model yolov8s --dataset Test-Images --samples 500 --output-dir ./results/yolov8s --quiet

echo "批量测试完成，结果保存在 ./results/ 目录"
```

### 输出文件
- `benchmark_log_timestamp.log` - 详细执行日志
- `modeltype_detailed_timestamp.csv` - 每帧详细时间数据
- `modeltype_summary_timestamp.csv` - 汇总统计信息
- `modeltype_speed_analysis_timestamp.png` - 速度分析图表
- `modeltype_summary_timestamp.png` - 性能总结图表

## 模型名称对照表

### 分类模型
| 命令行名称 | 模型描述 |
|------------|----------|
| `resnet18` | ResNet-18 |
| `resnet50` | ResNet-50 |
| `efficientnet_b0` | EfficientNet-B0 |
| `efficientnet_b3` | EfficientNet-B3 |
| `vit_base_patch16_224` | Vision Transformer Base |
| `mobilenetv3_large_100` | MobileNet-V3 Large |

### 检测模型
| 命令行名称 | 模型描述 |
|------------|----------|
| `yolov8n.pt` | YOLOv8 Nano |
| `yolov8s.pt` | YOLOv8 Small |
| `yolov8m.pt` | YOLOv8 Medium |
| `fasterrcnn-resnet50-fpn` | Faster R-CNN ResNet50 |
| `fasterrcnn-mobilenet-v3-large-fpn` | Faster R-CNN MobileNet |
| `fcos-resnet50-fpn` | FCOS ResNet50 |

### 分割模型
| 命令行名称 | 模型描述 |
|------------|----------|
| `deeplabv3plus_resnet50` | DeepLabV3+ ResNet50 |
| `deeplabv3plus_efficientnet_b0` | DeepLabV3+ EfficientNet-B0 |
| `unet_resnet34` | UNet ResNet34 |
| `unetplusplus_resnet50` | UNet++ ResNet50 |
| `pspnet_resnet50` | PSPNet ResNet50 |
| `fpn_resnet50` | FPN ResNet50 |


## 使用场景

### 1. 模型选择和比较
```bash
# 比较不同分类模型在相同数据集上的性能
python main.py --device cuda:0 --model-type classification --model resnet18 --dataset CIFAR-10 --samples 1000 --output-dir ./compare/resnet18
python main.py --device cuda:0 --model-type classification --model efficientnet_b0 --dataset CIFAR-10 --samples 1000 --output-dir ./compare/efficientnet_b0
```

### 2. 硬件性能测试
```bash
# 测试相同模型在不同设备上的性能
python main.py --device cpu --model-type classification --model resnet50 --dataset MNIST --samples 500 --output-dir ./hardware/cpu
python main.py --device cuda:0 --model-type classification --model resnet50 --dataset MNIST --samples 500 --output-dir ./hardware/gpu
```

### 3. 自动化CI/CD集成
```bash
# 在CI/CD管道中进行性能回归测试
python main.py --device cuda:0 --model-type detection --model yolov8n --dataset Test-Images --samples 100 --quiet --no-plots --output-dir ./ci_results
```

## 扩展指南

### 添加新模型
1. 在 `config.py` 中添加模型配置
2. 在 `models.py` 中实现加载逻辑
3. 更新 `cli.py` 中的模型验证逻辑
4. 必要时在 `rendering.py` 中添加渲染支持

### 添加新数据集
1. 在 `datasets.py` 中创建数据集类
2. 在 `DatasetLoader` 中添加加载方法
3. 在 `interactive.py` 和 `cli.py` 中添加选择选项

### 自定义评估指标
1. 在 `benchmarks.py` 中修改测试逻辑
2. 在 `monitoring.py` 中添加统计计算
3. 在 `output.py` 中更新可视化

## 注意事项

1. **内存管理**: 大规模测试时注意GPU内存使用
2. **依赖检查**: 程序会自动检查依赖，缺失的库会影响部分功能
3. **数据路径**: 确保数据集路径配置正确
4. **权限问题**: 某些系统可能需要管理员权限进行GPU监控
5. **命令行转义**: 在shell脚本中使用时注意特殊字符转义

## 故障排除

### 常见问题
1. **CUDA不可用** - 检查PyTorch CUDA安装
2. **模型下载失败** - 检查网络连接和存储空间
3. **内存不足** - 减少测试样本数量或使用更小的模型
4. **GPU监控失败** - 安装nvidia-ml-py3库
5. **命令行参数错误** - 使用 `--list-models` 查看可用模型名称

### 调试方法
1. 查看详细日志文件
2. 使用 `--list-models` 和 `--list-datasets` 检查可用选项
3. 先使用小样本数量测试
4. 检查依赖安装状态

## 贡献指南

欢迎提交问题报告和功能请求。在贡献代码时，请确保：
1. 遵循现有的代码风格
2. 添加适当的文档和注释
3. 进行充分的测试
4. 更新相关文档

## 许可证

本项目采用MIT许可证，详见LICENSE文件。