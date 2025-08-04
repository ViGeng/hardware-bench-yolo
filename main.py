#!/usr/bin/env python3
"""
Deep Learning Benchmark Tool - 深度学习基准测试工具
主程序入口模块

新增功能：
1. KITTI数据集支持
2. Faster R-CNN和FCOS目标检测模型
3. 语义分割模式(Segmentation)，包含DeepLab、PSPNet、UNet等模型
4. 完整的日志记录系统，包含时间戳和详细的执行状态
5. 模块化架构，便于维护和扩展
6. 命令行参数支持，可进行批量测试和自动化

主要功能：
1. 交互式设备选择 (CPU/GPU) - 支持返回上一级
2. 模型类型选择 (Detection/Classification/Segmentation) - 支持返回上一级
3. 数据集选择 (MNIST/CIFAR-10/COCO/ImageNet/KITTI/Cityscapes) - 支持返回上一级
4. 自定义样本数量选择 - 从100到全部数据集
5. 详细性能统计和CSV输出 - 按帧/图像输出详细时间信息
6. 结果可视化
7. 完整的日志记录系统
8. 命令行参数支持和批量测试

命令行使用示例：
python main.py --device cuda:0 --model-type classification --model resnet18 --dataset MNIST --samples 100
python main.py --device cpu --model-type detection --model yolov8n --dataset Test-Images --samples 500
python main.py --list-models

**列出所有可用模型**
```bash
python main.py --list-models
```

**列出所有可用数据集**
```bash
python main.py --list-datasets
```

Platform: Ubuntu 22.04 + NVIDIA GPU support
"""

import sys
import time
import logging
import os

# 导入自定义模块
from utils import setup_logging, check_dependencies, print_dependency_status
from interactive import InteractiveInterface
from cli import CommandLineInterface
from datasets import DatasetLoader
from models import ModelLoader
from rendering import RenderingEngine
from benchmarks import BenchmarkRunner
from monitoring import ResourceMonitor, StatisticsCalculator
from output import ResultExporter, Visualizer

class BenchmarkManager:
    """基准测试管理器 - 主要控制类"""
    
    def __init__(self):
        # 设置日志系统
        self.logger, self.log_filename = setup_logging()
        self.logger.info("基准测试工具初始化开始")
        
        # 检查依赖
        self.dependencies = check_dependencies()
        
        # 初始化界面组件
        self.interactive_interface = InteractiveInterface(self.dependencies)
        self.cli_interface = CommandLineInterface(self.dependencies)
        
        # 初始化其他组件
        self.resource_monitor = ResourceMonitor()
        self.stats_calculator = StatisticsCalculator()
        
        # 基准测试相关对象
        self.dataset_loader = None
        self.model_loader = None
        self.rendering_engine = None
        self.benchmark_runner = None
        
        # 运行数据
        self.configuration = None
        self.model = None
        self.dataloader = None
        self.test_images = None
        self.cli_mode = False
        
        self.logger.info("基准测试工具初始化完成")
    
    def run(self):
        """运行主程序"""
        try:
            # 解析命令行参数
            parser = self.cli_interface.create_parser()
            args = parser.parse_args()
            
            # 处理特殊命令
            if args.list_models:
                self.cli_interface.list_available_models()
                return
            
            if args.list_datasets:
                self.cli_interface.list_available_datasets()
                return
            
            # 判断使用交互模式还是命令行模式
            if self.cli_interface.should_use_interactive_mode(args):
                self._run_interactive_mode()
            else:
                self._run_cli_mode(args)
            
        except KeyboardInterrupt:
            self.logger.warning("程序被用户中断")
            print("\n程序被用户中断")
        except Exception as e:
            self.logger.error(f"程序运行过程中出错: {e}")
            print(f"程序运行过程中出错: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _run_interactive_mode(self):
        """运行交互模式"""
        self.logger.info("启动交互模式")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 深度学习基准测试工具启动")
        
        # 打印依赖状态
        print_dependency_status(self.dependencies)
        
        # 交互式设置
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始交互式配置")
        if not self.interactive_interface.run_interactive_setup():
            self.logger.info("用户取消设置，程序退出")
            return
        
        self.configuration = self.interactive_interface.get_configuration()
        self.cli_mode = False
        
        # 运行基准测试
        self._run_benchmark_pipeline()
    
    def _run_cli_mode(self, args):
        """运行命令行模式"""
        self.logger.info("启动命令行模式")
        
        # 验证参数
        errors = self.cli_interface.validate_args(args)
        if errors:
            print("参数错误:")
            for error in errors:
                print(f"  - {error}")
            print("\n使用 --help 查看帮助信息")
            sys.exit(1)
        
        # 转换为配置对象
        self.configuration = self.cli_interface.args_to_config(args)
        self.cli_mode = True
        
        # 创建输出目录
        if not os.path.exists(self.configuration['output_dir']):
            os.makedirs(self.configuration['output_dir'])
            
        # 打印配置摘要
        self.cli_interface.print_config_summary(self.configuration)
        
        # 运行基准测试
        self._run_benchmark_pipeline()
    
    def _run_benchmark_pipeline(self):
        """运行基准测试流程"""
        # 初始化组件
        self._initialize_components()
        
        # 加载数据集
        self._load_dataset()
        
        # 加载模型
        self._load_model()
        
        # 运行基准测试
        self._run_benchmark()
    
    def _initialize_components(self):
        """初始化各个组件"""
        self.logger.info("初始化各个组件")
        
        # 初始化数据集加载器
        self.dataset_loader = DatasetLoader(self.configuration['test_samples'])
        
        # 初始化模型加载器
        self.model_loader = ModelLoader(self.configuration['device'])
        
        # 初始化渲染引擎
        self.rendering_engine = RenderingEngine(self.logger)
        
        self.logger.info("组件初始化完成")
    
    def _load_dataset(self):
        """加载数据集"""
        self.logger.info("开始加载数据集")
        if not self.configuration.get('quiet', False):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始加载数据集")
        
        dataset_name = self.configuration['dataset_name']
        model_type = self.configuration['model_type']
        
        try:
            if model_type == 'classification':
                if dataset_name == 'MNIST':
                    self.dataloader = self.dataset_loader.load_mnist()
                elif dataset_name == 'CIFAR-10':
                    self.dataloader = self.dataset_loader.load_cifar10()
                elif dataset_name == 'ImageNet-Sample':
                    self.dataloader = self.dataset_loader.create_synthetic_classification_dataset()
                else:
                    raise ValueError(f"Unknown classification dataset: {dataset_name}")
            
            elif model_type == 'detection':
                if dataset_name == 'KITTI':
                    self.dataloader = self.dataset_loader.load_kitti()
                elif dataset_name in ['COCO-Sample', 'Test-Images']:
                    self.dataloader, self.test_images = self.dataset_loader.create_synthetic_detection_dataset()
                else:
                    raise ValueError(f"Unknown detection dataset: {dataset_name}")
            
            elif model_type == 'segmentation':
                if dataset_name == 'Cityscapes':
                    self.dataloader = self.dataset_loader.load_cityscapes()
                elif dataset_name == 'Synthetic-Segmentation':
                    self.dataloader = self.dataset_loader.create_synthetic_segmentation_dataset()
                else:
                    raise ValueError(f"Unknown segmentation dataset: {dataset_name}")
            
            self.logger.info("数据集加载成功")
            
        except Exception as e:
            self.logger.error(f"数据集加载失败: {e}")
            print(f"数据集加载失败: {e}")
            raise e
    
    def _load_model(self):
        """加载模型"""
        self.logger.info("开始加载模型")
        if not self.configuration.get('quiet', False):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始加载模型")
        
        try:
            self.model = self.model_loader.load_model(
                self.configuration['model_type'],
                self.configuration['model_info']
            )
            self.logger.info("模型加载成功")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            print(f"模型加载失败: {e}")
            raise e
    
    def _run_benchmark(self):
        """运行基准测试"""
        self.logger.info("开始运行基准测试")
        if not self.configuration.get('quiet', False):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始运行基准测试")
        
        # 启动资源监控
        monitor_thread = self.resource_monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            # 创建基准测试运行器
            self.benchmark_runner = BenchmarkRunner(
                model=self.model,
                model_type=self.configuration['model_type'],
                model_info=self.configuration['model_info'],
                device=self.configuration['device'],
                rendering_engine=self.rendering_engine,
                test_samples=self.configuration['test_samples']
            )
            
            # 运行对应类型的基准测试
            if self.configuration['model_type'] == 'classification':
                timing_results = self.benchmark_runner.run_classification_benchmark(self.dataloader)
            elif self.configuration['model_type'] == 'detection':
                timing_results = self.benchmark_runner.run_detection_benchmark(self.dataloader, self.test_images)
            elif self.configuration['model_type'] == 'segmentation':
                timing_results = self.benchmark_runner.run_segmentation_benchmark(self.dataloader)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            self.logger.info(f"基准测试完成，总耗时: {total_time:.2f}秒")
            
            # 停止资源监控
            self.resource_monitor.stop_monitoring()
            
            # 获取资源统计
            resource_stats = self.resource_monitor.get_resource_stats()
            
            # 计算统计信息
            stats = self.stats_calculator.calculate_benchmark_statistics(
                timing_results=timing_results,
                total_time=total_time,
                total_samples=self.benchmark_runner.total_samples,
                model_type=self.configuration['model_type'],
                model_info=self.configuration['model_info'],
                dataset_name=self.configuration['dataset_name'],
                device=self.configuration['device'],
                resource_stats=resource_stats
            )
            
            # 打印结果
            if not self.configuration.get('quiet', False):
                self.stats_calculator.print_results_summary(stats)
            
            # 保存结果和生成可视化
            self._save_results_and_visualizations(stats)
            
        except KeyboardInterrupt:
            self.logger.warning("测试被用户中断")
            print("\n测试被用户中断")
            return None
        except Exception as e:
            self.logger.error(f"测试过程中出错: {e}")
            print(f"测试过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.resource_monitor.stop_monitoring()
    
    def _save_results_and_visualizations(self, stats):
        """保存结果和生成可视化"""
        self.logger.info("开始保存结果和生成可视化")
        
        try:
            # 创建结果导出器
            exporter = ResultExporter(self.benchmark_runner.detailed_results)
            
            # 保存CSV结果到指定目录
            if self.cli_mode and 'output_dir' in self.configuration:
                # 在命令行模式下，保存到指定目录
                original_dir = os.getcwd()
                os.chdir(self.configuration['output_dir'])
            
            csv_filenames = exporter.save_detailed_csv_results(stats, self.configuration['model_type'])
            
            # 创建可视化（如果不是禁用状态）
            plot_files = []
            if not self.configuration.get('no_plots', False):
                visualizer = Visualizer(self.benchmark_runner.detailed_results)
                plot_files = visualizer.create_visualizations(stats, self.configuration['model_type'])
            
            # 恢复原目录（如果改变了的话）
            if self.cli_mode and 'output_dir' in self.configuration:
                os.chdir(original_dir)
                # 更新文件路径为绝对路径
                output_dir = os.path.abspath(self.configuration['output_dir'])
                csv_filenames = [os.path.join(output_dir, os.path.basename(f)) for f in csv_filenames]
                plot_files = [os.path.join(output_dir, os.path.basename(f)) for f in plot_files]
            
            # 打印最终结果文件信息
            if not self.configuration.get('quiet', False):
                print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 测试完成!")
                if self.cli_mode:
                    print(f"结果保存在: {self.configuration['output_dir']}")
                print(f"日志文件: {self.log_filename}")
                print(f"详细结果文件: {csv_filenames[0]}")
                print(f"汇总结果文件: {csv_filenames[1]}")
                
                if plot_files:
                    print("生成的图表文件:")
                    for plot_file in plot_files:
                        print(f"  - {plot_file}")
                elif self.configuration.get('no_plots', False):
                    print("图表生成已禁用")
            
            # 记录最终完成状态到日志
            self.logger.info("所有测试和输出生成完成")
            log_msg = f"生成文件: 日志-{self.log_filename}, CSV-{len(csv_filenames)}个, 图表-{len(plot_files)}个"
            self.logger.info(log_msg)
            
            # 在命令行模式下，提供简洁的成功信息
            if self.cli_mode and self.configuration.get('quiet', False):
                print(f"SUCCESS: Results saved to {self.configuration['output_dir']}")
                print(f"Throughput: {stats['performance']['throughput']:.2f} samples/sec")
                print(f"Rating: {stats['performance'].get('rating', 'N/A')}")
            
        except Exception as e:
            self.logger.error(f"保存结果时出错: {e}")
            print(f"保存结果时出错: {e}")

def main():
    """主函数入口"""
    benchmark_manager = BenchmarkManager()
    benchmark_manager.run()

if __name__ == "__main__":
    main()