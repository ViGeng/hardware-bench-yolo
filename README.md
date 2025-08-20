# 深度学习基准测试工具

支持分类、目标检测和语义分割任务的综合基准测试工具。

## 功能特点

- **多种模型类型**: 支持分类、目标检测和语义分割模型
- **多种数据集**: 内置支持MNIST、CIFAR-10、COCO、KITTI、Cityscapes等数据集
- **综合监控**: 实时CPU、内存和GPU使用率跟踪
- **详细分析**: 逐样本时间分解和统计分析
- **可视化**: 自动生成性能图表
- **灵活接口**: 交互式和命令行界面
- **导出选项**: 结果导出为CSV和可视化图表


### 基础安装
```bash
pip install dl-benchmark-tool

### 完整安装（包含所有可选依赖）
```bash
pip install dl-benchmark-tool[full]

## 目录

dl-benchmark-tool/
├── setup.py
├── pyproject.toml
├── README.md
├── LICENSE
├── MANIFEST.in
├── requirements.txt
├── dl_benchmark/
│   ├── __init__.py
│   ├── main.py
│   ├── monitoring.py
│   ├── output.py
│   ├── rendering.py
│   ├── utils.py
│   ├── benchmarks.py
│   ├── cli.py
│   ├── config.py
│   ├── datasets.py
│   ├── interactive.py
│   └── models.py
└── tests/
    ├── __init__.py
    ├── test_basic.py
    └── test_sample.py




### Building and Publishing Steps

1. Prepare the package:
```bash
cd dl-benchmark-tool
pip install build twine

2Build the package:
```bash
python -m build

.Test the package locally:
```bash
pip install dist/dl_benchmark_tool-0.1.0-py3-none-any.whl

4.Run tests:
```bash
python -m pytest tests/ -v

5.Upload to PyPI (test first):
```bash
# Test PyPI first
python -m twine upload --repository testpypi dist/*

# Then real PyPI
python -m twine upload dist/*