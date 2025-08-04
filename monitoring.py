#!/usr/bin/env python3
"""
监控模块 - 负责系统资源监控和统计计算
"""

import time
import threading
import logging
import socket
from collections import deque
import numpy as np
import torch
import psutil
from utils import get_system_info, calculate_performance_rating, check_dependencies

# 检查依赖
dependencies = check_dependencies()

class ResourceMonitor:
    """系统资源监控器"""
    
    def __init__(self, enable_gpu_monitoring=True):
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        # 监控数据存储
        self.cpu_usage = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.gpu_memory = deque(maxlen=1000)
        self.gpu_utilization = deque(maxlen=1000)
        
        # GPU监控相关
        self.nvml_available = dependencies['pynvml'] and enable_gpu_monitoring
        self.gpu_handle = None
        
        if torch.cuda.is_available() and self.nvml_available:
            self._try_initialize_gpu_monitoring()
        elif not enable_gpu_monitoring:
            pass  # 静默，不记录日志
        elif not torch.cuda.is_available():
            pass  # 静默，不记录日志  
        elif not dependencies['pynvml']:
            pass  # 静默，不记录日志
    
    def _try_initialize_gpu_monitoring(self):
        """尝试初始化GPU监控，包括PATH修复"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # 只有成功时才记录日志
            self.logger.info("GPU详细监控已启用")
            return
        except pynvml.NVMLError_LibraryNotFound:
            # 静默尝试修复PATH问题
            if self._fix_nvidia_path():
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    self.logger.info("GPU详细监控已启用")
                    return
                except Exception:
                    pass  # 静默失败
            
            # 不显示详细的失败信息，GPU监控不是核心功能
            self.gpu_handle = None
        except Exception:
            # 其他异常也静默处理
            self.gpu_handle = None
    
    def _fix_nvidia_path(self):
        """尝试修复NVIDIA路径问题"""
        import os
        import glob
        
        # Windows上常见的NVIDIA路径
        base_nvidia_paths = [
            r"C:\Program Files\NVIDIA Corporation\NVSMI",
            r"C:\Windows\System32",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.*\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.*\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.*\bin",
        ]
        
        # 展开通配符路径
        nvidia_paths = []
        for base_path in base_nvidia_paths:
            if '*' in base_path:
                nvidia_paths.extend(glob.glob(base_path))
            else:
                nvidia_paths.append(base_path)
        
        # 静默搜索nvidia-smi.exe
        for drive in ['C:']:
            search_pattern = f"{drive}\\Program Files*\\NVIDIA*\\**\\nvidia-smi.exe"
            try:
                found_files = glob.glob(search_pattern, recursive=True)
                for found_file in found_files:
                    nvsmi_dir = os.path.dirname(found_file)
                    if nvsmi_dir not in nvidia_paths:
                        nvidia_paths.append(nvsmi_dir)
            except:
                pass
        
        current_path = os.environ.get('PATH', '')
        added_paths = []
        
        for nvidia_path in nvidia_paths:
            if os.path.exists(nvidia_path) and nvidia_path not in current_path:
                os.environ['PATH'] = current_path + os.pathsep + nvidia_path
                current_path = os.environ['PATH']
                added_paths.append(nvidia_path)
        
        # 只有在添加了路径时才返回True，不记录详细日志
        return len(added_paths) > 0
    
    def start_monitoring(self):
        """开始资源监控"""
        self.logger.info("开始系统资源监控")
        self.monitoring = True
        
        monitor_thread = threading.Thread(target=self._monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return monitor_thread
    
    def stop_monitoring(self):
        """停止资源监控"""
        self.monitoring = False
        time.sleep(0.5)  # 等待监控线程结束
        self.logger.info("系统资源监控结束")
    
    def _monitor_resources(self):
        """监控系统资源"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU和内存
                self.cpu_usage.append(process.cpu_percent(interval=0.1))
                self.memory_usage.append(psutil.virtual_memory().percent)
                
                # GPU监控
                if torch.cuda.is_available():
                    # GPU内存
                    gpu_mem = torch.cuda.memory_allocated(0)
                    gpu_total = torch.cuda.get_device_properties(0).total_memory
                    self.gpu_memory.append((gpu_mem / gpu_total) * 100)
                    
                    # GPU利用率
                    if self.gpu_handle and self.nvml_available:
                        try:
                            import pynvml
                            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                            self.gpu_utilization.append(util.gpu)
                        except:
                            self.gpu_utilization.append(0)
                
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"资源监控过程中出错: {e}")
                break
    
    def get_resource_stats(self):
        """获取资源使用统计"""
        stats = {
            'cpu': {
                'min': np.min(self.cpu_usage) if self.cpu_usage else 0,
                'max': np.max(self.cpu_usage) if self.cpu_usage else 0,
                'avg': np.mean(self.cpu_usage) if self.cpu_usage else 0,
                'std': np.std(self.cpu_usage) if self.cpu_usage else 0
            },
            'memory': {
                'min': np.min(self.memory_usage) if self.memory_usage else 0,
                'max': np.max(self.memory_usage) if self.memory_usage else 0,
                'avg': np.mean(self.memory_usage) if self.memory_usage else 0,
                'std': np.std(self.memory_usage) if self.memory_usage else 0
            }
        }
        
        # GPU统计
        if torch.cuda.is_available() and self.gpu_memory:
            stats['gpu'] = {
                'memory': {
                    'min': np.min(self.gpu_memory),
                    'max': np.max(self.gpu_memory),
                    'avg': np.mean(self.gpu_memory),
                    'std': np.std(self.gpu_memory)
                },
                'utilization': {
                    'min': np.min(self.gpu_utilization) if self.gpu_utilization else 0,
                    'max': np.max(self.gpu_utilization) if self.gpu_utilization else 0,
                    'avg': np.mean(self.gpu_utilization) if self.gpu_utilization else 0,
                    'std': np.std(self.gpu_utilization) if self.gpu_utilization else 0
                }
            }
        
        return stats

class StatisticsCalculator:
    """统计计算器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_benchmark_statistics(self, timing_results, total_time, total_samples, 
                                     model_type, model_info, dataset_name, device, resource_stats):
        """计算基准测试统计信息"""
        self.logger.info("开始计算统计信息")
        
        # 获取系统信息
        system_info = get_system_info()
        
        stats = {
            'system_info': {
                'hostname': system_info['hostname'],
                'device': device,
                'model_type': model_type,
                'model_name': model_info['name'],
                'dataset': dataset_name,
                'torch_version': system_info['torch_version'],
                'cuda_available': system_info['cuda_available'],
                'device_name': system_info['device_name']
            },
            'performance': {
                'total_samples': total_samples,
                'total_time': total_time,
                'throughput': total_samples / total_time if total_time > 0 else 0,
                'avg_time_per_sample': (total_time / total_samples * 1000) if total_samples > 0 else 0
            },
            'timing': {},
            'resources': resource_stats
        }
        
        # 添加时间统计
        for key, times in timing_results.items():
            if times:
                stats['timing'][key] = {
                    'min': np.min(times),
                    'max': np.max(times),
                    'avg': np.mean(times),
                    'std': np.std(times)
                }
        
        # 计算性能评级
        fps = stats['performance']['throughput']
        performance_rating = calculate_performance_rating(model_type, fps)
        stats['performance']['rating'] = performance_rating
        
        # 记录关键统计信息到日志
        self.logger.info(f"统计信息计算完成 - 吞吐量: {stats['performance']['throughput']:.2f} samples/sec")
        self.logger.info(f"性能评级: {performance_rating}, FPS: {fps:.2f}")
        
        return stats
    
    def print_results_summary(self, stats):
        """打印简洁的结果总结"""
        self.logger.info("开始打印测试结果")
        
        print("\n" + "="*70)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*70)
        
        # 基本信息
        print(f"Model: {stats['system_info']['model_name']}")
        print(f"Dataset: {stats['system_info']['dataset']}")
        print(f"Device: {stats['system_info']['device']}")
        print(f"Device Name: {stats['system_info']['device_name']}")
        
        # 性能指标
        print(f"\n{'='*20} PERFORMANCE METRICS {'='*20}")
        print(f"  Samples processed: {stats['performance']['total_samples']}")
        print(f"  Total time: {stats['performance']['total_time']:.2f}s")
        print(f"  Throughput: {stats['performance']['throughput']:.2f} samples/sec")
        print(f"  Avg time/sample: {stats['performance']['avg_time_per_sample']:.2f}ms")
        
        # 时间分解
        if stats['timing']:
            print(f"\n{'='*20} TIMING BREAKDOWN (ms) {'='*20}")
            for stage, data in stats['timing'].items():
                stage_name = stage.replace('_', ' ').title()
                print(f"  {stage_name}:")
                print(f"    Min: {data['min']:.2f}ms")
                print(f"    Max: {data['max']:.2f}ms")
                print(f"    Avg: {data['avg']:.2f}ms ± {data['std']:.2f}")
                print()
        
        # 资源使用
        print(f"{'='*20} RESOURCE UTILIZATION {'='*20}")
        print(f"  CPU Usage:")
        print(f"    Min: {stats['resources']['cpu']['min']:.1f}%")
        print(f"    Max: {stats['resources']['cpu']['max']:.1f}%")
        print(f"    Avg: {stats['resources']['cpu']['avg']:.1f}% ± {stats['resources']['cpu']['std']:.1f}")
        print()
        
        print(f"  Memory Usage:")
        print(f"    Min: {stats['resources']['memory']['min']:.1f}%")
        print(f"    Max: {stats['resources']['memory']['max']:.1f}%")
        print(f"    Avg: {stats['resources']['memory']['avg']:.1f}% ± {stats['resources']['memory']['std']:.1f}")
        print()
        
        if 'gpu' in stats['resources']:
            print(f"  GPU Memory:")
            print(f"    Min: {stats['resources']['gpu']['memory']['min']:.1f}%")
            print(f"    Max: {stats['resources']['gpu']['memory']['max']:.1f}%")
            print(f"    Avg: {stats['resources']['gpu']['memory']['avg']:.1f}% ± {stats['resources']['gpu']['memory']['std']:.1f}")
            print()
            
            print(f"  GPU Utilization:")
            print(f"    Min: {stats['resources']['gpu']['utilization']['min']:.1f}%")
            print(f"    Max: {stats['resources']['gpu']['utilization']['max']:.1f}%")
            print(f"    Avg: {stats['resources']['gpu']['utilization']['avg']:.1f}% ± {stats['resources']['gpu']['utilization']['std']:.1f}")
            print()
        
        # 性能评级
        print(f"{'='*20} OVERALL RATING {'='*25}")
        print(f"  Performance Rating: {stats['performance']['rating']}")
        print("="*70)