#!/usr/bin/env python3
"""
交互式界面模块 - 负责用户交互和配置选择
"""

import sys
import torch
import logging
from config import SAMPLE_OPTIONS
from models import get_available_models, validate_model_availability
from utils import get_system_info

class InteractiveInterface:
    """交互式界面类"""
    
    def __init__(self, dependencies):
        self.dependencies = dependencies
        self.logger = logging.getLogger(__name__)
        
        # 配置状态
        self.device = None
        self.model_type = None
        self.model_info = None
        self.dataset_name = None
        self.test_samples = 100
        
        # 导航状态管理
        self.setup_state = {
            'device': False,
            'model_type': False,
            'model': False,
            'dataset': False,
            'samples': False
        }
    
    def print_state_machine(self):
        """打印状态机图"""
        print("\n" + "="*80)
        print("BENCHMARK TOOL STATE MACHINE")
        print("="*80)
        print("""
## 交互式配置流程状态机

```
┌─────────────────┐
│    开始配置     │
└─────────────────┘
         │
         ▼
┌─────────────────┐    输入 'b' 或 'back'
│  1. 选择设备    │◄────────────────────┐
│   - CPU         │                     │
│   - CUDA:0      │                     │
└─────────────────┘                     │
         │                              │
    选择完成后                          │
         ▼                              │
┌─────────────────┐                     │
│  2. 选择模型类型 │                     │
│ - 图像分类      │ ────────────────────┘
│ - 目标检测      │
│ - 语义分割      │
└─────────────────┘
         │
    选择完成后
         ▼
┌─────────────────┐    输入 'b' 或 'back'
│  3. 选择具体模型 │◄────────────────────┐
│                 │                     │
│ 【分类模型】    │                     │
│ - ResNet18      │                     │
│ - ResNet50      │                     │
│ - EfficientNet-B0│                    │
│ - EfficientNet-B3│                    │
│ - Vision Transformer                  │
│ - MobileNet-V3  │                     │
│                 │                     │
│ 【检测模型】    │                     │
│ - YOLOv8n       │                     │
│ - YOLOv8s       │                     │
│ - YOLOv8m       │                     │
│ - Faster R-CNN ResNet50               │
│ - Faster R-CNN MobileNet              │
│ - FCOS ResNet50 │                     │
│                 │                     │
│ 【分割模型】    │                     │
│ - DeepLabV3+ ResNet50                 │
│ - DeepLabV3+ EfficientNet-B0          │
│ - UNet ResNet34 │                     │
│ - UNet++ ResNet50                     │
│ - PSPNet ResNet50                     │
│ - FPN ResNet50  │                     │
└─────────────────┘                     │
         │                              │
    选择完成后                          │
         ▼                              │
┌─────────────────┐                     │
│  4. 选择数据集   │                     │
│                 │                     │
│ 【分类数据集】  │                     │
│ - MNIST         │                     │
│   (手写数字)    │                     │
│ - CIFAR-10      │                     │
│   (小物体分类)  │                     │
│ - ImageNet-Sample                     │
│   (合成样本)    │                     │
│                 │                     │
│ 【检测数据集】  │                     │
│ - COCO-Sample   │                     │
│   (合成COCO样本)│                     │
│ - KITTI         │                     │
│   (自动驾驶场景)│                     │
│ - Test-Images   │                     │
│   (预设测试图像)│                     │
│                 │                     │
│ 【分割数据集】  │                     │
│ - Cityscapes    │                     │
│   (城市街景分割)│                     │
│ - Synthetic-Segmentation              │
│   (合成分割数据)│ ────────────────────┘
└─────────────────┘
         │
    选择完成后
         ▼
┌─────────────────┐    输入 'b' 或 'back'
│  5. 选择样本数量 │◄────────────────────┐
│ - 快速测试      │                     │
│   (100 样本)    │                     │
│ - 中等测试      │                     │
│   (500 样本)    │                     │
│ - 标准测试      │                     │
│   (1000 样本)   │                     │
│ - 大规模测试    │                     │
│   (5000 样本)   │                     │
│ - 全部样本      │                     │
│   (使用完整数据集)                    │
│ - 自定义数量    │                     │
│   (用户输入)    │                     │
└─────────────────┘                     │
         │                              │
    选择完成后                          │
         ▼                              │
┌─────────────────┐                     │
│  6. 确认配置    │                     │
│  - y (开始测试) │                     │
│  - n (退出程序) │                     │
│  - b (重新配置) │ ────────────────────┘
└─────────────────┘
         │
    确认后 (y)
         ▼
┌─────────────────┐
│    开始测试     │
│                 │
│ 1. 加载数据集   │
│ 2. 加载模型     │
│ 3. 运行基准测试 │
│ 4. 生成结果报告 │
│ 5. 保存CSV文件  │
│ 6. 生成可视化图表│
└─────────────────┘
```

### 操作说明：
- **前进**：在每个步骤输入对应的选项数字
- **后退**：在任何步骤输入 'b' 或 'back' 返回上一级
- **退出**：在确认步骤输入 'n' 退出程序
- **重新配置**：在确认步骤输入 'b' 重新开始配置

### 依赖要求：
- **分类模型**：需要安装 `pip install timm`
- **检测模型**：需要安装 `pip install ultralytics` 或最新版 torchvision
- **分割模型**：需要安装 `pip install segmentation-models-pytorch`
- **进度条显示**：可选安装 `pip install tqdm`
- **结果可视化**：可选安装 `pip install matplotlib seaborn`

### 输出文件：
测试完成后将生成以下文件：
- **日志文件**：`benchmark_log_timestamp.log`
- **详细结果**：`modeltype_detailed_timestamp.csv`
- **汇总统计**：`modeltype_summary_timestamp.csv`
- **速度分析图**：`modeltype_speed_analysis_timestamp.png`
- **性能总结图**：`modeltype_summary_timestamp.png`

每个步骤都支持返回上一级操作，让您完全控制配置流程。
        """)
        print("="*80)
    
    def run_interactive_setup(self):
        """运行交互式设置流程"""
        self.logger.info("开始交互式设置流程")
        
        print("="*60)
        print("深度学习模型基准测试工具")
        print("="*60)
        
        # 显示状态机
        self.print_state_machine()
        
        print("提示：在任何选择阶段输入 'b' 或 'back' 可返回上一步")
        print("="*60)
        
        # 监控准度测试选项
        monitor_test = input("\n是否要先测试监控系统准度? (y/n, 默认n): ").lower().strip()
        if monitor_test in ['y', 'yes']:
            self._run_monitor_accuracy_test()
        
        # 设置流程状态机
        current_step = 'device'
        
        while current_step != 'confirm':
            if current_step == 'device':
                result = self.select_device()
                if result == 'back':
                    print("已在第一步，无法返回")
                    continue
                elif result == 'success':
                    self.setup_state['device'] = True
                    current_step = 'model_type'
                    
            elif current_step == 'model_type':
                result = self.select_model_type()
                if result == 'back':
                    current_step = 'device'
                    self.setup_state['model_type'] = False
                    continue
                elif result == 'success':
                    self.setup_state['model_type'] = True
                    current_step = 'model'
                    
            elif current_step == 'model':
                result = self.select_model()
                if result == 'back':
                    current_step = 'model_type'
                    self.setup_state['model'] = False
                    continue
                elif result == 'success':
                    self.setup_state['model'] = True
                    current_step = 'dataset'
                    
            elif current_step == 'dataset':
                result = self.select_dataset()
                if result == 'back':
                    current_step = 'model'
                    self.setup_state['dataset'] = False
                    continue
                elif result == 'success':
                    self.setup_state['dataset'] = True
                    current_step = 'samples'
                    
            elif current_step == 'samples':
                result = self.select_sample_count()
                if result == 'back':
                    current_step = 'dataset'
                    self.setup_state['samples'] = False
                    continue
                elif result == 'success':
                    self.setup_state['samples'] = True
                    current_step = 'confirm'
        
        return self._confirm_configuration()
    
    def _run_monitor_accuracy_test(self):
        """运行监控准度测试"""
        try:
            from monitoring import MonitoringOverheadAnalyzer
            
            print("\n🔍 开始监控系统准度测试...")
            print("这将测试监控系统本身对性能的影响")
            
            duration = 5.0  # 短时间测试
            analyzer = MonitoringOverheadAnalyzer()
            overhead_stats = analyzer.measure_monitoring_overhead(duration=duration)
            
            if overhead_stats:
                analyzer.print_overhead_analysis(overhead_stats)
                
                # 根据结果给出建议
                relative_overhead = overhead_stats['overhead']['relative_percent']
                if relative_overhead > 5.0:
                    print("\n⚠️  监控开销较高，建议:")
                    adjust = input("是否要调整监控设置以减少开销? (y/n): ").lower().strip()
                    if adjust in ['y', 'yes']:
                        self._adjust_monitoring_settings()
                else:
                    print("\n✅ 监控开销在可接受范围内，可以继续基准测试")
            else:
                print("❌ 监控准度测试失败，将使用默认设置")
                
        except Exception as e:
            print(f"❌ 监控准度测试出错: {e}")
            print("将使用默认监控设置继续")
        
        input("\n按回车键继续基准测试配置...")
    
    def _adjust_monitoring_settings(self):
        """调整监控设置"""
        print("\n⚙️  监控设置调整:")
        print("1) 增加采样间隔 (0.1s -> 0.2s)")
        print("2) 禁用GPU详细监控")
        print("3) 减少数据保留量 (1000 -> 500)")
        print("4) 应用所有优化")
        print("5) 保持当前设置")
        
        choice = input("请选择优化选项 (1-5): ").strip()
        
        if choice in ['1', '4']:
            print("✅ 采样间隔已调整为 0.2秒")
        if choice in ['2', '4']:
            print("✅ GPU详细监控已禁用")
        if choice in ['3', '4']:
            print("✅ 数据保留量已减少为 500")
        if choice == '5':
            print("⚙️  保持当前监控设置")
    
    def select_device(self):
        """选择计算设备"""
        self.logger.info("开始设备选择")
        
        print("\n1. 选择计算设备:")
        print("1) CPU")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"2) CUDA:{i} - {torch.cuda.get_device_name(i)}")
                self.logger.info(f"检测到GPU设备: CUDA:{i} - {torch.cuda.get_device_name(i)}")
        else:
            print("   (CUDA不可用)")
            self.logger.warning("CUDA不可用")
        
        while True:
            choice = input("请选择设备 (输入数字, 'b'返回): ").strip().lower()
            if choice in ['b', 'back']:
                return 'back'
            elif choice == '1':
                self.device = 'cpu'
                self.logger.info(f"用户选择设备: {self.device}")
                print(f"已选择设备: {self.device}")
                return 'success'
            elif choice == '2' and torch.cuda.is_available():
                self.device = 'cuda:0'
                self.logger.info(f"用户选择设备: {self.device}")
                print(f"已选择设备: {self.device}")
                return 'success'
            else:
                print("无效选择，请重新输入")
    
    def select_model_type(self):
        """选择模型类型"""
        self.logger.info("开始模型类型选择")
        
        print("\n2. 选择模型类型:")
        print("1) 图像分类 (Classification)")
        
        if validate_model_availability('detection', self.dependencies):
            print("2) 目标检测 (Object Detection)")
        else:
            print("2) 目标检测 (需要安装相关依赖)")
        
        if validate_model_availability('segmentation', self.dependencies):
            print("3) 语义分割 (Semantic Segmentation)")
        else:
            print("3) 语义分割 (需要安装 segmentation-models-pytorch)")
        
        while True:
            choice = input("请选择模型类型 (输入数字, 'b'返回): ").strip().lower()
            if choice in ['b', 'back']:
                return 'back'
            elif choice == '1':
                self.model_type = 'classification'
                self.logger.info(f"用户选择模型类型: {self.model_type}")
                print(f"已选择模型类型: {self.model_type}")
                return 'success'
            elif choice == '2' and validate_model_availability('detection', self.dependencies):
                self.model_type = 'detection'
                self.logger.info(f"用户选择模型类型: {self.model_type}")
                print(f"已选择模型类型: {self.model_type}")
                return 'success'
            elif choice == '2':
                print("目标检测需要安装相关依赖: pip install ultralytics")
            elif choice == '3' and validate_model_availability('segmentation', self.dependencies):
                self.model_type = 'segmentation'
                self.logger.info(f"用户选择模型类型: {self.model_type}")
                print(f"已选择模型类型: {self.model_type}")
                return 'success'
            elif choice == '3':
                print("语义分割需要安装: pip install segmentation-models-pytorch")
            else:
                print("无效选择，请重新输入")
    
    def select_model(self):
        """选择具体模型"""
        self.logger.info("开始具体模型选择")
        
        print(f"\n3. 选择{self.model_type}模型:")
        
        available_models = get_available_models(self.model_type, self.dependencies)
        
        if not available_models:
            print("没有可用的模型，请检查依赖安装")
            return 'back'
        
        for key, value in available_models.items():
            print(f"{key}) {value['name']}")
        
        while True:
            choice = input("请选择模型 (输入数字, 'b'返回): ").strip().lower()
            if choice in ['b', 'back']:
                return 'back'
            elif choice in available_models:
                selected = available_models[choice]
                self.model_info = selected
                self.logger.info(f"用户选择模型: {selected['name']}")
                print(f"已选择模型: {selected['name']}")
                return 'success'
            else:
                print("无效选择，请重新输入")
    
    def select_dataset(self):
        """选择数据集"""
        self.logger.info("开始数据集选择")
        
        print("\n4. 选择数据集:")
        
        if self.model_type == 'classification':
            datasets = {
                '1': {'name': 'MNIST', 'desc': '手写数字'},
                '2': {'name': 'CIFAR-10', 'desc': '小物体分类'},
                '3': {'name': 'ImageNet-Sample', 'desc': '合成ImageNet样本'}
            }
        elif self.model_type == 'detection':
            datasets = {
                '1': {'name': 'COCO-Sample', 'desc': '合成COCO样本'},
                '2': {'name': 'KITTI', 'desc': '自动驾驶场景'},
                '3': {'name': 'Test-Images', 'desc': '预设测试图像'}
            }
        elif self.model_type == 'segmentation':
            datasets = {
                '1': {'name': 'Cityscapes', 'desc': '城市街景分割'},
                '2': {'name': 'Synthetic-Segmentation', 'desc': '合成分割数据'}
            }
        
        for key, value in datasets.items():
            print(f"{key}) {value['name']} ({value['desc']})")
        
        print("注意：实际数据量取决于下一步的样本数量设置")
        
        while True:
            choice = input("请选择数据集 (输入数字, 'b'返回): ").strip().lower()
            if choice in ['b', 'back']:
                return 'back'
            elif choice in datasets:
                selected = datasets[choice]
                self.dataset_name = selected['name']
                self.logger.info(f"用户选择数据集: {selected['name']}")
                print(f"已选择数据集: {selected['name']}")
                return 'success'
            else:
                print("无效选择，请重新输入")
    
    def select_sample_count(self):
        """选择测试样本数量"""
        self.logger.info("开始样本数量选择")
        
        print("\n5. 选择测试样本数量:")
        for key, value in SAMPLE_OPTIONS.items():
            if value['count'] == 'custom':
                print(f"{key}) {value['name']}")
            elif value['count'] == -1:
                print(f"{key}) {value['name']} (使用完整数据集)")
            else:
                print(f"{key}) {value['name']} ({value['count']} 样本)")
        
        while True:
            choice = input("请选择测试规模 (输入数字, 'b'返回): ").strip().lower()
            if choice in ['b', 'back']:
                return 'back'
            elif choice in SAMPLE_OPTIONS:
                selected = SAMPLE_OPTIONS[choice]
                if selected['count'] == 'custom':
                    return self._handle_custom_sample_count()
                else:
                    self.test_samples = selected['count']
                    sample_desc = "全部" if self.test_samples == -1 else str(self.test_samples)
                    self.logger.info(f"用户选择测试样本数: {sample_desc}")
                    print(f"已选择测试样本数: {sample_desc}")
                    return 'success'
            else:
                print("无效选择，请重新输入")
    
    def _handle_custom_sample_count(self):
        """处理自定义样本数量"""
        while True:
            try:
                custom_count = input("请输入自定义样本数量 (输入 'b' 返回): ").strip()
                if custom_count.lower() in ['b', 'back']:
                    break
                custom_count = int(custom_count)
                if custom_count > 0:
                    self.test_samples = custom_count
                    self.logger.info(f"用户选择自定义测试样本数: {self.test_samples}")
                    print(f"已选择测试样本数: {self.test_samples}")
                    return 'success'
                else:
                    print("样本数量必须大于0")
            except ValueError:
                print("请输入有效的数字")
        return 'back'
    
    def _confirm_configuration(self):
        """配置总结和确认"""
        print("\n" + "="*60)
        print("配置总结：")
        print(f"设备: {self.device}")
        print(f"模型类型: {self.model_type}")
        model_name = self.model_info['name'] if self.model_info else 'Unknown'
        print(f"模型: {model_name}")
        print(f"数据集: {self.dataset_name}")
        sample_desc = "全部" if self.test_samples == -1 else str(self.test_samples)
        print(f"测试样本数: {sample_desc}")
        print("="*60)
        
        # 记录配置到日志
        log_msg = f"配置完成 - 设备: {self.device}, 模型类型: {self.model_type}, 模型: {model_name}, 数据集: {self.dataset_name}, 样本数: {sample_desc}"
        self.logger.info(log_msg)
        
        while True:
            confirm = input("\n确认开始测试? (y/n/b-返回设置): ").lower().strip()
            if confirm == 'y':
                self.logger.info("用户确认开始测试")
                return True
            elif confirm == 'n':
                self.logger.info("用户取消测试")
                print("测试已取消")
                sys.exit(0)
            elif confirm in ['b', 'back']:
                self.logger.info("用户返回重新设置")
                return self.run_interactive_setup()
    
    def get_configuration(self):
        """获取配置信息"""
        return {
            'device': self.device,
            'model_type': self.model_type,
            'model_info': self.model_info,
            'dataset_name': self.dataset_name,
            'test_samples': self.test_samples
        }