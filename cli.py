#!/usr/bin/env python3
"""
命令行界面模块 - 处理命令行参数和非交互式执行
"""

import argparse
import sys
import torch
import logging
from config import DETECTION_MODELS, CLASSIFICATION_MODELS, SEGMENTATION_MODELS, SAMPLE_OPTIONS
from utils import check_dependencies
from models import validate_model_availability

class CommandLineInterface:
    """命令行界面类"""
    
    def __init__(self, dependencies):
        self.dependencies = dependencies
        self.logger = logging.getLogger(__name__)
    
    def create_parser(self):
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(
            description='深度学习模型基准测试工具',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例用法:
  # 交互模式（默认）
  python main.py
  
  # 使用CPU测试ResNet18分类模型
  python main.py --device cpu --model-type classification --model resnet18 --dataset MNIST --samples 100
  
  # 使用GPU测试YOLOv8检测模型
  python main.py --device cuda:0 --model-type detection --model yolov8n --dataset Test-Images --samples 500
  
  # 使用GPU测试分割模型
  python main.py --device cuda:0 --model-type segmentation --model unet_resnet34 --dataset Synthetic-Segmentation --samples 200
  
  # 列出可用模型
  python main.py --list-models
            """
        )
        
        # 添加各种命令行参数
        parser.add_argument('--device', 
                          choices=['cpu', 'cuda:0', 'auto'], 
                          default='auto',
                          help='计算设备 (default: auto)')
        
        parser.add_argument('--model-type', 
                          choices=['classification', 'detection', 'segmentation'],
                          help='模型类型')
        
        parser.add_argument('--model',
                          help='模型名称（使用 --list-models 查看可用模型）')
        
        parser.add_argument('--dataset',
                          help='数据集名称')
        
        parser.add_argument('--samples', 
                          type=int, 
                          default=100,
                          help='测试样本数量 (default: 100, -1 表示全部)')
        
        parser.add_argument('--batch-size',
                          type=int,
                          default=1,
                          help='批处理大小 (default: 1)')
        
        parser.add_argument('--output-dir',
                          default='./results',
                          help='输出目录 (default: ./results)')
        
        parser.add_argument('--no-plots',
                          action='store_true',
                          help='不生成可视化图表')
        
        parser.add_argument('--quiet',
                          action='store_true',
                          help='静默模式，减少输出')
        
        parser.add_argument('--list-models',
                          action='store_true',
                          help='列出所有可用模型')
        
        parser.add_argument('--list-datasets',
                          action='store_true',
                          help='列出所有可用数据集')
        
        parser.add_argument('--interactive',
                          action='store_true',
                          help='强制使用交互模式')
        
        return parser
    
    def list_available_models(self):
        """列出所有可用模型"""
        print("可用模型列表：")
        print("="*60)
        
        # 分类模型
        print("\n图像分类模型 (Classification):")
        if validate_model_availability('classification', self.dependencies):
            for key, model in CLASSIFICATION_MODELS.items():
                status = "✓" if self.dependencies['timm'] else "✗"
                print(f"  {status} {model['model']:<25} - {model['name']}")
        else:
            print("  需要安装: pip install timm")
        
        # 检测模型
        print("\n目标检测模型 (Detection):")
        if validate_model_availability('detection', self.dependencies):
            for key, model in DETECTION_MODELS.items():
                if model['type'] == 'yolo':
                    status = "✓" if self.dependencies['ultralytics'] else "✗"
                    req = "ultralytics" if not self.dependencies['ultralytics'] else ""
                    # 显示两种格式：带.pt和不带.pt
                    model_name = model['model']
                    if model_name.endswith('.pt'):
                        model_id = f"{model_name} 或 {model_name[:-3]}"
                    else:
                        model_id = model_name
                elif model['type'] == 'torchvision':
                    status = "✓" if self.dependencies['torchvision_detection'] else "✗"
                    req = "torchvision (最新版)" if not self.dependencies['torchvision_detection'] else ""
                    model_id = model['model'].replace('_', '-')
                else:
                    status = "✗"
                    req = "未知依赖"
                    model_id = model['model']
                
                print(f"  {status} {model_id:<30} - {model['name']}")
                if req:
                    print(f"    需要安装: pip install {req}")
        else:
            print("  需要安装: pip install ultralytics 或更新 torchvision")
        
        # 分割模型
        print("\n语义分割模型 (Segmentation):")
        if validate_model_availability('segmentation', self.dependencies):
            for key, model in SEGMENTATION_MODELS.items():
                status = "✓" if self.dependencies['smp'] else "✗"
                model_id = f"{model['model'].lower()}_{model['encoder'].replace('-', '_')}"
                print(f"  {status} {model_id:<25} - {model['name']}")
        else:
            print("  需要安装: pip install segmentation-models-pytorch")
    
    def list_available_datasets(self):
        """列出所有可用数据集"""
        print("可用数据集列表：")
        print("="*60)
        
        print("\n分类数据集:")
        print("  MNIST              - 手写数字识别 (28x28 -> 224x224)")
        print("  CIFAR-10           - 小物体分类 (32x32 -> 224x224)")
        print("  ImageNet-Sample    - 合成ImageNet样本 (224x224)")
        
        print("\n检测数据集:")
        print("  COCO-Sample        - 合成COCO样本")
        print("  KITTI              - 自动驾驶场景数据")
        print("  Test-Images        - 预设测试图像")
        
        print("\n分割数据集:")
        print("  Cityscapes         - 城市街景分割")
        print("  Synthetic-Segmentation - 合成分割数据")
    
    def validate_args(self, args):
        """验证命令行参数"""
        errors = []
        
        # 如果指定了模型类型，必须指定模型和数据集
        if args.model_type and not args.model:
            errors.append("指定了 --model-type 但未指定 --model")
        
        if args.model_type and not args.dataset:
            errors.append("指定了 --model-type 但未指定 --dataset")
        
        # 验证设备
        if args.device == 'cuda:0' and not torch.cuda.is_available():
            errors.append("指定了CUDA设备但CUDA不可用")
        
        # 验证模型类型的可用性
        if args.model_type:
            if not validate_model_availability(args.model_type, self.dependencies):
                if args.model_type == 'classification':
                    errors.append("分类模型不可用，需要安装: pip install timm")
                elif args.model_type == 'detection':
                    errors.append("检测模型不可用，需要安装: pip install ultralytics")
                elif args.model_type == 'segmentation':
                    errors.append("分割模型不可用，需要安装: pip install segmentation-models-pytorch")
        
        # 验证模型名称
        if args.model_type and args.model:
            valid_model = self._validate_model_name(args.model_type, args.model)
            if not valid_model:
                errors.append(f"无效的模型名称: {args.model}")
        
        # 验证数据集名称
        if args.model_type and args.dataset:
            valid_dataset = self._validate_dataset_name(args.model_type, args.dataset)
            if not valid_dataset:
                errors.append(f"无效的数据集名称: {args.dataset}")
        
        # 验证样本数量
        if args.samples < -1 or args.samples == 0:
            errors.append("样本数量必须是正数或-1（表示全部）")
        
        return errors
    
    def _validate_model_name(self, model_type, model_name):
        """验证模型名称是否有效"""
        if model_type == 'classification':
            valid_models = [model['model'] for model in CLASSIFICATION_MODELS.values()]
            return model_name in valid_models
        
        elif model_type == 'detection':
            valid_models = []
            for model in DETECTION_MODELS.values():
                if model['type'] == 'yolo':
                    # YOLO模型支持带.pt和不带.pt的格式
                    valid_models.append(model['model'])
                    if model['model'].endswith('.pt'):
                        valid_models.append(model['model'][:-3])  # 去掉.pt后缀
                else:
                    # torchvision模型使用下划线格式
                    valid_models.append(model['model'])
                    valid_models.append(model['model'].replace('_', '-'))
            return model_name in valid_models
        
        elif model_type == 'segmentation':
            valid_models = []
            for model in SEGMENTATION_MODELS.values():
                model_id = f"{model['model'].lower()}_{model['encoder'].replace('-', '_')}"
                valid_models.append(model_id)
            return model_name in valid_models
        
        return False
    
    def _validate_dataset_name(self, model_type, dataset_name):
        """验证数据集名称是否有效"""
        valid_datasets = {
            'classification': ['MNIST', 'CIFAR-10', 'ImageNet-Sample'],
            'detection': ['COCO-Sample', 'KITTI', 'Test-Images'],
            'segmentation': ['Cityscapes', 'Synthetic-Segmentation']
        }
        
        return dataset_name in valid_datasets.get(model_type, [])
    
    def args_to_config(self, args):
        """将命令行参数转换为配置对象"""
        # 自动选择设备
        if args.device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        
        # 查找模型信息
        model_info = self._find_model_info(args.model_type, args.model)
        
        config = {
            'device': device,
            'model_type': args.model_type,
            'model_info': model_info,
            'dataset_name': args.dataset,
            'test_samples': args.samples,
            'batch_size': args.batch_size,
            'output_dir': args.output_dir,
            'no_plots': args.no_plots,
            'quiet': args.quiet
        }
        
        return config
    
    def _find_model_info(self, model_type, model_name):
        """根据模型名称查找模型信息"""
        if model_type == 'classification':
            for model in CLASSIFICATION_MODELS.values():
                if model['model'] == model_name:
                    return model
        
        elif model_type == 'detection':
            for model in DETECTION_MODELS.values():
                if model['type'] == 'yolo':
                    # 支持 yolov8n 或 yolov8n.pt 两种格式
                    if model['model'] == model_name or model['model'] == f"{model_name}.pt":
                        return model
                    if model['model'].endswith('.pt') and model['model'][:-3] == model_name:
                        return model
                elif model['type'] == 'torchvision':
                    if model['model'] == model_name or model['model'].replace('_', '-') == model_name:
                        return model
        
        elif model_type == 'segmentation':
            for model in SEGMENTATION_MODELS.values():
                model_id = f"{model['model'].lower()}_{model['encoder'].replace('-', '_')}"
                if model_id == model_name:
                    return model
        
        return None
    
    def should_use_interactive_mode(self, args):
        """判断是否应该使用交互模式"""
        # 如果显式指定交互模式
        if args.interactive:
            return True
        
        # 如果指定了list命令
        if args.list_models or args.list_datasets:
            return False
        
        # 如果没有指定足够的参数，使用交互模式
        if not (args.model_type and args.model and args.dataset):
            return True
        
        return False
    
    def print_config_summary(self, config):
        """打印配置摘要"""
        if not config.get('quiet', False):
            print("\n" + "="*60)
            print("基准测试配置:")
            print("="*60)
            print(f"设备: {config['device']}")
            print(f"模型类型: {config['model_type']}")
            print(f"模型: {config['model_info']['name']}")
            print(f"数据集: {config['dataset_name']}")
            print(f"样本数: {config['test_samples'] if config['test_samples'] != -1 else '全部'}")
            print(f"输出目录: {config['output_dir']}")
            print(f"生成图表: {'否' if config['no_plots'] else '是'}")
            print("="*60)