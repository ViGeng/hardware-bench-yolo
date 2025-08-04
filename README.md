# Deep Learning Benchmark Tool

一个深度学习模型基准测试工具，支持分类、检测和分割模型的性能评估。支持交互式界面和命令行参数两种使用方式。

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
├── README.md            # 说明文档
└── changelog.txt        # 更新日志
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