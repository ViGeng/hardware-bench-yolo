#!/usr/bin/env python3
"""
YOLOv8 性能基准测试工具 - 改进版
主要改进：
1. 支持命令行参数，不用修改代码
2. 支持多种输出格式（txt, json, csv）
3. 更好的错误处理
4. 自动检测文件和设备
5. 可选显示检测结果（像原版一样）
6. 可选显示进度信息
7. 完整的可视化功能（英文标签，清晰图表）
"""

import argparse
import json
import csv
from ultralytics import YOLO
import numpy as np
import time
import psutil
import torch
import threading
import socket
from collections import deque
import os
import sys
from pathlib import Path

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available, GPU monitoring disabled")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
    # 设置字体和样式 - 使用英文避免中文字体问题
    plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.autolayout'] = True
    sns.set_style("whitegrid")
    sns.set_palette("husl")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available, visualization disabled")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv8 Performance Benchmark Tool')
    
    # 主要参数
    parser.add_argument('--model', '-m', default='yolov8n.pt',
                        help='模型路径或名称 (默认: yolov8n.pt)')
    parser.add_argument('--source', '-s', default='0',
                        help='视频源: 文件路径或摄像头设备号 (默认: 0)')
    
    # 可选参数
    parser.add_argument('--device', '-d', default='auto',
                        help='运行设备: cpu, cuda:0, auto (默认: auto)')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='批次大小 (默认: 1)')
    parser.add_argument('--max-frames', '-f', type=int,
                        help='最大处理帧数 (默认: 无限制)')
    parser.add_argument('--imgsz', type=int,
                        help='输入图像尺寸 (默认: 模型默认值)')
    
    # 输出选项
    parser.add_argument('--output-format', '-o', nargs='+', 
                        choices=['txt', 'json', 'csv'], default=['txt'],
                        help='输出格式 (默认: txt)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细信息')
    parser.add_argument('--show-detections', action='store_true',
                        help='显示每帧检测结果')
    parser.add_argument('--show-progress', action='store_true',
                        help='显示处理进度（每10帧）')
    parser.add_argument('--plot', action='store_true',
                        help='生成可视化图表')
    parser.add_argument('--plot-realtime', action='store_true',
                        help='实时显示性能曲线')
    
    return parser.parse_args()

class BenchmarkTool:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.monitoring = True
        
        # 系统信息
        self.cpu_count = psutil.cpu_count()
        self.cpu_logical_count = psutil.cpu_count(logical=True)
        
        # 初始化指标收集
        self.preprocess_times = []
        self.inference_times = []
        self.postprocess_times = []
        self.total_frames = 0
        self.start_time = None
        
        # 初始化资源监控
        self.cpu_percentages = deque(maxlen=1000)
        self.memory_usages = deque(maxlen=1000)
        self.gpu_mem_usages = deque(maxlen=1000) if torch.cuda.is_available() else None
        self.gpu_util_usages = deque(maxlen=1000) if torch.cuda.is_available() else None
        
        # 可视化数据收集
        self.frame_timestamps = []
        self.frame_fps = []
        self.frame_inference_times = []
        self.resource_timestamps = []
        
        self.monitor_thread = None
        self.nvml_handle = None
    
    def print_detection_results(self, result, frame_num):
        """显示检测结果"""
        try:
            if result.boxes is not None and len(result.boxes) > 0:
                # 获取检测到的类别
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                
                # 统计每种物品的数量
                detections = {}
                for cls_id, conf in zip(class_ids, confidences):
                    class_name = result.names[cls_id]
                    if class_name not in detections:
                        detections[class_name] = []
                    detections[class_name].append(conf)
                
                # 格式化输出
                if detections:
                    detection_strs = []
                    for class_name, confs in detections.items():
                        count = len(confs)
                        avg_conf = sum(confs) / count
                        if count == 1:
                            detection_strs.append(f"{class_name}({avg_conf:.2f})")
                        else:
                            detection_strs.append(f"{class_name}x{count}({avg_conf:.2f})")
                    
                    print(f"Frame {frame_num}: {', '.join(detection_strs)}")
                else:
                    print(f"Frame {frame_num}: No objects detected")
            else:
                print(f"Frame {frame_num}: No objects detected")
                
        except Exception as e:
            if self.args.verbose:
                print(f"Frame {frame_num}: Detection display error: {e}")
            else:
                print(f"Frame {frame_num}: -")
        
    def load_model(self):
        """加载YOLO模型"""
        print(f"正在加载模型: {self.args.model}")
        try:
            self.model = YOLO(self.args.model)
            print(f"模型加载成功")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)
    
    def validate_source(self):
        """验证输入源"""
        if self.args.source.isdigit():
            print(f"使用摄像头设备: {self.args.source}")
            return int(self.args.source)
        elif os.path.exists(self.args.source):
            print(f"使用视频文件: {self.args.source}")
            return self.args.source
        else:
            print(f"错误: 找不到输入源 '{self.args.source}'")
            sys.exit(1)
    
    def check_and_fix_device(self):
        """检查并修复设备设置"""
        if self.args.device == 'auto':
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                # 测试CUDA是否真的可用
                try:
                    test_tensor = torch.tensor([1.0]).cuda()
                    self.args.device = 'cuda:0'
                    print(f"自动选择设备: cuda:0")
                except Exception as e:
                    print(f"CUDA不可用，自动切换到CPU: {e}")
                    self.args.device = 'cpu'
            else:
                print("未检测到可用的CUDA设备，使用CPU")
                self.args.device = 'cpu'
        
        # 验证设备
        if 'cuda' in self.args.device:
            try:
                test_tensor = torch.tensor([1.0]).to(self.args.device)
                print(f"设备验证成功: {self.args.device}")
            except Exception as e:
                print(f"设备 {self.args.device} 不可用，切换到CPU: {e}")
                self.args.device = 'cpu'

    def monitor_resources(self):
        """资源监控函数"""
        # 初始化GPU监控
        if torch.cuda.is_available() and NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                if self.args.verbose:
                    print("GPU监控已初始化")
            except Exception as e:
                if self.args.verbose:
                    print(f"GPU监控初始化失败: {e}")
                self.nvml_handle = None
        
        # 获取当前进程
        current_process = psutil.Process()
        
        while self.monitoring:
            try:
                current_time = time.time()
                
                # CPU使用率
                self.cpu_percentages.append(current_process.cpu_percent(interval=0.1))
                
                # 内存使用率
                memory = psutil.virtual_memory()
                self.memory_usages.append(memory.percent)
                
                # 记录时间戳用于可视化
                if self.start_time:
                    self.resource_timestamps.append(current_time - self.start_time)
                
                # GPU使用率
                if torch.cuda.is_available():
                    # GPU内存使用率
                    gpu_mem_alloc = torch.cuda.memory_allocated(0)
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory
                    gpu_mem_percent = (gpu_mem_alloc / gpu_mem_total) * 100
                    self.gpu_mem_usages.append(gpu_mem_percent)
                    
                    # GPU利用率
                    if self.nvml_handle is not None:
                        try:
                            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                            self.gpu_util_usages.append(utilization.gpu)
                        except Exception:
                            last_value = self.gpu_util_usages[-1] if self.gpu_util_usages else 0
                            self.gpu_util_usages.append(last_value)
                
                time.sleep(0.1)
                
            except Exception as e:
                if self.args.verbose:
                    print(f"监控错误: {e}")
                break
        
        # 清理
        if torch.cuda.is_available() and self.nvml_handle is not None and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
    
    def run_benchmark(self):
        """运行基准测试"""
        print("开始基准测试...")
        
        # 检查并修复设备设置
        self.check_and_fix_device()
        
        # 启动资源监控线程
        self.monitor_thread = threading.Thread(target=self.monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # 验证输入源
        source = self.validate_source()
        
        # 设置推理参数
        inference_params = {
            'stream': True,
            'batch': self.args.batch_size,
            'device': self.args.device,
            'verbose': False  # 避免干扰输出
        }
        
        if self.args.imgsz:
            inference_params['imgsz'] = self.args.imgsz
        
        if self.args.verbose:
            print(f"推理参数: {inference_params}")
            print(f"CPU配置: {self.cpu_count}核心/{self.cpu_logical_count}线程 (最大CPU使用率: {self.cpu_logical_count * 100}%)")
        
        self.start_time = time.time()
        
        # 用于计算瞬时FPS
        last_time = self.start_time
        last_frame_count = 0
        
        try:
            # 运行推理
            results = self.model(source, **inference_params)
            
            # 处理结果
            for result in results:
                # 检查帧数限制（如果不想限制帧数，可以注释掉下面几行）
                if self.args.max_frames and self.total_frames >= self.args.max_frames:
                    if self.args.verbose:
                        print(f"达到最大帧数限制: {self.args.max_frames}")
                    break
                
                # 收集指标
                speed_data = result.speed
                self.preprocess_times.append(speed_data.get("preprocess", 0))
                self.inference_times.append(speed_data.get("inference", 0))
                self.postprocess_times.append(speed_data.get("postprocess", 0))
                self.total_frames += 1
                
                # 收集可视化数据
                current_time = time.time()
                self.frame_timestamps.append(current_time - self.start_time)
                
                # 计算当前FPS
                if len(self.frame_timestamps) >= 2:
                    time_diff = self.frame_timestamps[-1] - self.frame_timestamps[-2]
                    current_fps = 1.0 / time_diff if time_diff > 0 else 0
                else:
                    current_fps = 0
                self.frame_fps.append(current_fps)
                self.frame_inference_times.append(speed_data.get("inference", 0))
                
                # 显示检测结果（如果启用）
                if self.args.show_detections:
                    self.print_detection_results(result, self.total_frames)
                
                # 显示进度（显示累积平均FPS和瞬时FPS）
                if (self.args.show_progress or self.args.verbose) and self.total_frames % 10 == 0:
                    current_time = time.time()
                    
                    # 累积平均FPS
                    avg_fps = self.total_frames / (current_time - self.start_time)
                    
                    # 瞬时FPS（最近10帧的平均）
                    instant_fps = 10 / (current_time - last_time) if current_time > last_time else 0
                    
                    print(f"已处理 {self.total_frames} 帧 | 平均FPS: {avg_fps:.2f} | 瞬时FPS: {instant_fps:.2f}")
                    
                    last_time = current_time
                    last_frame_count = self.total_frames
                
        except KeyboardInterrupt:
            print("\n测试被用户中断")
        except Exception as e:
            print(f"推理过程出错: {e}")
            return False
        
        # 停止监控
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        if self.total_frames == 0:
            print("没有处理任何帧")
            return False
        
        print(f"测试完成，共处理 {self.total_frames} 帧")
        return True
    
    def calculate_statistics(self):
        """计算统计数据"""
        if not self.preprocess_times:
            return None
        
        end_time = time.time()
        total_time = end_time - self.start_time
        throughput = self.total_frames / total_time if total_time > 0 else 0
        
        total_per_frame = [p + i + pp for p, i, pp in zip(
            self.preprocess_times, self.inference_times, self.postprocess_times)]
        
        stats = {
            'summary': {
                'total_frames': self.total_frames,
                'total_time': total_time,
                'throughput': throughput,
                'avg_frame_time': np.mean(total_per_frame)
            },
            'timing': {
                'preprocess': {
                    'min': np.min(self.preprocess_times),
                    'max': np.max(self.preprocess_times),
                    'avg': np.mean(self.preprocess_times),
                    'std': np.std(self.preprocess_times)
                },
                'inference': {
                    'min': np.min(self.inference_times),
                    'max': np.max(self.inference_times),
                    'avg': np.mean(self.inference_times),
                    'std': np.std(self.inference_times)
                },
                'postprocess': {
                    'min': np.min(self.postprocess_times),
                    'max': np.max(self.postprocess_times),
                    'avg': np.mean(self.postprocess_times),
                    'std': np.std(self.postprocess_times)
                },
                'total_per_frame': {
                    'min': np.min(total_per_frame),
                    'max': np.max(total_per_frame),
                    'avg': np.mean(total_per_frame),
                    'std': np.std(total_per_frame)
                }
            },
            'resources': {
                'cpu': {
                    'min': np.min(self.cpu_percentages) if self.cpu_percentages else 0,
                    'max': np.max(self.cpu_percentages) if self.cpu_percentages else 0,
                    'avg': np.mean(self.cpu_percentages) if self.cpu_percentages else 0
                },
                'memory': {
                    'min': np.min(self.memory_usages) if self.memory_usages else 0,
                    'max': np.max(self.memory_usages) if self.memory_usages else 0,
                    'avg': np.mean(self.memory_usages) if self.memory_usages else 0
                }
            },
            'system_info': {
                'hostname': socket.gethostname(),
                'model': self.args.model,
                'source': self.args.source,
                'device': self.args.device,
                'batch_size': self.args.batch_size,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'cpu_cores': self.cpu_count,
                'cpu_threads': self.cpu_logical_count,
                'cpu_max_usage': f"{self.cpu_logical_count * 100}%"
            }
        }
        
        # 添加GPU统计
        if torch.cuda.is_available() and self.gpu_mem_usages and self.gpu_util_usages:
            # 确保有足够的数据
            gpu_mem_data = list(self.gpu_mem_usages)
            gpu_util_data = list(self.gpu_util_usages)
            
            if len(gpu_mem_data) > 0 and len(gpu_util_data) > 0:
                stats['resources']['gpu'] = {
                    'memory': {
                        'min': float(np.min(gpu_mem_data)),
                        'max': float(np.max(gpu_mem_data)),
                        'avg': float(np.mean(gpu_mem_data))
                    },
                    'utilization': {
                        'min': float(np.min(gpu_util_data)),
                        'max': float(np.max(gpu_util_data)),
                        'avg': float(np.mean(gpu_util_data))
                    }
                }
            else:
                # 如果数据不足，设置默认值
                stats['resources']['gpu'] = {
                    'memory': {'min': 0.0, 'max': 0.0, 'avg': 0.0},
                    'utilization': {'min': 0.0, 'max': 0.0, 'avg': 0.0}
                }
        else:
            # 如果GPU不可用或没有数据，不添加GPU统计或设置为None
            if self.args.verbose:
                print("   GPU数据不可用，跳过GPU统计")
        
        return stats
    
    def create_visualizations(self, stats, save_path=None):
        """创建可视化图表"""
        if not VISUALIZATION_AVAILABLE:
            print("❌ matplotlib/seaborn不可用，跳过可视化")
            return
        
        if not self.frame_timestamps:
            print("❌ 没有足够的数据用于可视化")
            return
        
        try:
            print("📊 正在生成可视化图表...")
            
            # 设置matplotlib后端
            import matplotlib
            matplotlib.use('Agg')  # 强制使用Agg后端，确保可以保存图片
            
            # 创建图表
            fig = plt.figure(figsize=(16, 12))
            
            print("   - 生成FPS曲线...")
            # 1. 性能曲线图
            plt.subplot(2, 3, 1)
            plt.plot(self.frame_timestamps, self.frame_fps, 'b-', linewidth=2, alpha=0.8, label='Actual FPS')
            plt.title('Real-time FPS Performance', fontsize=14, fontweight='bold')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frames Per Second (FPS)')
            plt.grid(True, alpha=0.3)
            avg_fps = np.mean(self.frame_fps[10:]) if len(self.frame_fps) > 10 else np.mean(self.frame_fps)
            plt.axhline(y=avg_fps, color='r', linestyle='--', alpha=0.8, 
                       label=f'Average: {avg_fps:.2f} FPS')
            
            # 添加性能区间参考线
            plt.axhline(y=30, color='g', linestyle=':', alpha=0.5, label='Excellent (30+ FPS)')
            plt.axhline(y=15, color='orange', linestyle=':', alpha=0.5, label='Good (15+ FPS)')
            plt.axhline(y=5, color='red', linestyle=':', alpha=0.5, label='Minimum (5+ FPS)')
            
            plt.legend(fontsize=8)
            plt.ylim(bottom=0)
            
            print("   - 生成时间分布图...")
            # 2. 推理时间分布
            plt.subplot(2, 3, 2)
            inference_times = self.frame_inference_times
            n_bins = min(30, max(10, len(inference_times) // 5))
            plt.hist(inference_times, bins=n_bins, alpha=0.7, color='skyblue', 
                    edgecolor='black', density=False)
            plt.title('Inference Time Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Inference Time (milliseconds)')
            plt.ylabel('Number of Frames')
            
            mean_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            plt.axvline(x=mean_time, color='r', linestyle='--', 
                       label=f'Mean: {mean_time:.2f}ms')
            plt.axvline(x=mean_time + std_time, color='orange', linestyle=':', 
                       label=f'+1σ: {mean_time + std_time:.2f}ms')
            plt.axvline(x=mean_time - std_time, color='orange', linestyle=':', 
                       label=f'-1σ: {mean_time - std_time:.2f}ms')
            plt.legend(fontsize=8)
            
            print("   - 生成时间分解图...")
            # 3. 处理时间分解
            plt.subplot(2, 3, 3)
            stages = ['Preprocess', 'Inference', 'Postprocess']
            times = [
                np.mean(self.preprocess_times),
                np.mean(self.inference_times), 
                np.mean(self.postprocess_times)
            ]
            colors = ['lightcoral', 'skyblue', 'lightgreen']
            bars = plt.bar(stages, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            plt.title('Processing Time Breakdown', fontsize=14, fontweight='bold')
            plt.ylabel('Time (milliseconds)')
            plt.xlabel('Processing Stage')
            
            # 添加数值标签和百分比
            total_time = sum(times)
            for bar, time_val in zip(bars, times):
                percentage = (time_val / total_time) * 100
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                        f'{time_val:.1f}ms\n({percentage:.1f}%)', 
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            plt.ylim(0, max(times) * 1.2)
            
            print("   - 生成资源使用图...")
            # 4. 资源使用率时间序列
            plt.subplot(2, 3, 4)
            if self.cpu_percentages and self.resource_timestamps:
                cpu_data = list(self.cpu_percentages)
                time_data = self.resource_timestamps[:len(cpu_data)]
                plt.plot(time_data, cpu_data, 'g-', label='CPU Usage', linewidth=2, alpha=0.8)
                
                if self.gpu_util_usages and torch.cuda.is_available():
                    gpu_data = list(self.gpu_util_usages)[:len(time_data)]
                    plt.plot(time_data, gpu_data, 'r-', label='GPU Utilization', linewidth=2, alpha=0.8)
                
                plt.title('Resource Usage Over Time', fontsize=14, fontweight='bold')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Usage Percentage (%)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, max(100, max(cpu_data) * 1.1) if cpu_data else 100)
            else:
                plt.text(0.5, 0.5, 'No Resource Data Available', 
                        ha='center', va='center', transform=plt.gca().transAxes,
                        fontsize=12, style='italic')
                plt.title('Resource Usage Over Time', fontsize=14, fontweight='bold')
            
            print("   - 生成资源对比图...")
            # 5. CPU vs GPU负载对比
            plt.subplot(2, 3, 5)
            resource_labels = ['CPU Usage\n(%)', 'GPU Utilization\n(%)', 'Memory Usage\n(%)']
            
            # 安全获取资源数据，避免KeyError
            cpu_avg = stats['resources']['cpu']['avg'] if 'cpu' in stats['resources'] and 'avg' in stats['resources']['cpu'] else 0
            gpu_avg = 0
            if 'gpu' in stats['resources'] and stats['resources']['gpu']:
                if 'utilization' in stats['resources']['gpu'] and 'avg' in stats['resources']['gpu']['utilization']:
                    gpu_avg = stats['resources']['gpu']['utilization']['avg']
            memory_avg = stats['resources']['memory']['avg'] if 'memory' in stats['resources'] and 'avg' in stats['resources']['memory'] else 0
            
            resource_values = [cpu_avg, gpu_avg, memory_avg]
            colors = ['lightcoral', 'skyblue', 'lightgreen']
            
            bars = plt.bar(resource_labels, resource_values, color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=1)
            plt.title('Average Resource Utilization', fontsize=14, fontweight='bold')
            plt.ylabel('Utilization Percentage (%)')
            
            # 添加数值标签和性能评估
            for bar, value, label in zip(bars, resource_values, ['CPU', 'GPU', 'Memory']):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(resource_values)*0.02,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                # 添加性能评估
                if label == 'CPU':
                    if value > 300:
                        status = 'High Load'
                    elif value > 100:
                        status = 'Multi-core'
                    else:
                        status = 'Normal'
                elif label == 'GPU':
                    if value > 80:
                        status = 'High Util'
                    elif value > 50:
                        status = 'Good Util'
                    else:
                        status = 'Low Util'
                else:  # Memory
                    if value > 80:
                        status = 'High Usage'
                    elif value > 50:
                        status = 'Moderate'
                    else:
                        status = 'Low Usage'
                
                plt.text(bar.get_x() + bar.get_width()/2, -max(resource_values)*0.08,
                        status, ha='center', va='top', fontsize=8, style='italic')
            
            plt.ylim(-max(resource_values)*0.1, max(resource_values)*1.15)
            
            print("   - 生成性能评估图...")
            # 6. 性能评估雷达图
            try:
                plt.subplot(2, 3, 6, projection='polar')
                
                # 安全获取数据，避免KeyError
                gpu_avg = 50  # 默认值
                if 'gpu' in stats['resources'] and stats['resources']['gpu']:
                    if 'utilization' in stats['resources']['gpu'] and 'avg' in stats['resources']['gpu']['utilization']:
                        gpu_avg = stats['resources']['gpu']['utilization']['avg']
                
                cpu_avg = stats['resources']['cpu']['avg'] if 'cpu' in stats['resources'] and 'avg' in stats['resources']['cpu'] else 100
                inference_avg = stats['timing']['inference']['avg'] if 'inference' in stats['timing'] and 'avg' in stats['timing']['inference'] else 50
                inference_std = stats['timing']['inference']['std'] if 'inference' in stats['timing'] and 'std' in stats['timing']['inference'] else 10
                throughput = stats['summary']['throughput'] if 'throughput' in stats['summary'] else 10
                
                # 性能指标 (归一化到0-100)
                metrics = {
                    'FPS\nPerformance': min(throughput / 30 * 100, 100),  # 30fps为满分
                    'Inference\nSpeed': max(0, 100 - inference_avg / 100 * 100),  # 100ms为0分
                    'CPU\nEfficiency': max(0, 100 - cpu_avg / 400 * 100),  # 400%为0分
                    'GPU\nUtilization': min(gpu_avg, 100),  # 直接使用GPU利用率
                    'Performance\nStability': max(0, 100 - inference_std / 50 * 100)  # 50ms标准差为0分
                }
                
                angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
                values = list(metrics.values())
                
                # 闭合图形
                angles = np.concatenate((angles, [angles[0]]))
                values.append(values[0])
                
                plt.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.8, markersize=6)
                plt.fill(angles, values, alpha=0.25, color='blue')
                
                # 设置标签
                labels = list(metrics.keys())
                plt.xticks(angles[:-1], labels, fontsize=9)
                
                # 设置刻度和网格
                plt.ylim(0, 100)
                plt.yticks([20, 40, 60, 80, 100], ['20', '40', '60', '80', '100'], fontsize=8)
                plt.grid(True, alpha=0.3)
                
                # 添加性能评级圆圈
                for radius, color, alpha in [(80, 'green', 0.1), (60, 'yellow', 0.1), (40, 'orange', 0.1)]:
                    circle_angles = np.linspace(0, 2*np.pi, 100)
                    circle_values = [radius] * 100
                    plt.plot(circle_angles, circle_values, color=color, alpha=alpha, linewidth=1)
                
                plt.title('Performance Radar Chart\n(Higher = Better)', fontsize=12, fontweight='bold', pad=20)
                
                # 添加图例说明
                legend_text = f"Overall Score: {np.mean(values[:-1]):.1f}/100"
                plt.text(0.5, -0.15, legend_text, transform=plt.gca().transAxes, 
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                
            except Exception as e:
                print(f"   - 雷达图生成失败，使用柱状图替代: {e}")
                # 如果雷达图失败，创建简单的柱状图替代
                plt.subplot(2, 3, 6)
                
                # 安全获取数据
                gpu_avg = 0
                if 'gpu' in stats['resources'] and stats['resources']['gpu']:
                    if 'utilization' in stats['resources']['gpu'] and 'avg' in stats['resources']['gpu']['utilization']:
                        gpu_avg = stats['resources']['gpu']['utilization']['avg']
                        
                cpu_avg = stats['resources']['cpu']['avg'] if 'cpu' in stats['resources'] and 'avg' in stats['resources']['cpu'] else 0
                memory_avg = stats['resources']['memory']['avg'] if 'memory' in stats['resources'] and 'avg' in stats['resources']['memory'] else 0
                throughput = stats['summary']['throughput'] if 'throughput' in stats['summary'] else 0
                
                metrics_simple = ['FPS', 'CPU%', 'GPU%', 'Memory%']
                values_simple = [throughput, cpu_avg, gpu_avg, memory_avg]
                colors_simple = ['blue', 'red', 'green', 'orange']
                
                bars = plt.bar(metrics_simple, values_simple, color=colors_simple, alpha=0.7, 
                              edgecolor='black', linewidth=1)
                plt.title('Performance Metrics', fontsize=14, fontweight='bold')
                plt.ylabel('Value / Percentage')
                plt.xlabel('Metric Type')
                
                # 添加数值标签
                for bar, value, metric in zip(bars, values_simple, metrics_simple):
                    unit = ' FPS' if metric == 'FPS' else '%'
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values_simple)*0.02,
                            f'{value:.1f}{unit}', ha='center', va='bottom', fontweight='bold')
                
                plt.ylim(0, max(values_simple) * 1.15)
            
            plt.tight_layout(pad=2.0)
            
            # 添加整体标题和说明
            fig.suptitle(f'YOLO Performance Analysis Report\n'
                        f'Model: {stats["system_info"]["model"]} | Device: {stats["system_info"]["device"]} | '
                        f'Frames: {stats["summary"]["total_frames"]} | Avg FPS: {stats["summary"]["throughput"]:.2f}',
                        fontsize=16, fontweight='bold', y=0.98)
            
            # 添加图表说明
            explanation = (
                "Chart Explanations:\n"
                "• Top Left: Real-time FPS shows performance consistency over time\n"
                "• Top Center: Inference time distribution shows processing variability\n" 
                "• Top Right: Processing breakdown shows time spent in each stage\n"
                "• Bottom Left: Resource usage shows CPU/GPU utilization over time\n"
                "• Bottom Center: Average resource utilization comparison\n"
                "• Bottom Right: Overall performance radar (higher values = better)"
            )
            
            plt.figtext(0.02, 0.02, explanation, fontsize=9, 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            
            # 保存图表
            if save_path:
                print(f"   - 保存图表至: {save_path}")
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                
                # 验证文件是否真的保存了
                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path)
                    print(f"   ✅ 图表保存成功! 大小: {file_size} bytes")
                    
                    # 创建性能摘要文本文件
                    summary_path = save_path.replace('_visualization.png', '_performance_summary.txt')
                    self.create_performance_summary(stats, summary_path)
                    
                else:
                    print(f"   ❌ 图表保存失败，文件不存在")
            
            # 关闭图表释放内存
            plt.close()
            
            return fig
            
        except Exception as e:
            print(f"❌ 创建可视化图表失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_performance_summary(self, stats, summary_path):
        """创建性能摘要说明文件"""
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("YOLO Performance Analysis Summary\n")
                f.write("="*50 + "\n\n")
                
                # 基本信息
                f.write("Basic Information:\n")
                f.write(f"Model: {stats['system_info']['model']}\n")
                f.write(f"Device: {stats['system_info']['device']}\n")
                f.write(f"Batch Size: {stats['system_info']['batch_size']}\n")
                f.write(f"Total Frames: {stats['summary']['total_frames']}\n")
                f.write(f"Test Duration: {stats['summary']['total_time']:.2f} seconds\n\n")
                
                # 性能指标解读
                f.write("Performance Metrics Explanation:\n")
                f.write("-" * 30 + "\n")
                
                fps = stats['summary']['throughput']
                f.write(f"Average FPS: {fps:.2f}\n")
                if fps >= 30:
                    f.write("  → Excellent: Suitable for real-time applications\n")
                elif fps >= 15:
                    f.write("  → Good: Suitable for most video processing tasks\n")
                elif fps >= 5:
                    f.write("  → Fair: Suitable for offline processing\n")
                else:
                    f.write("  → Poor: Consider optimization or hardware upgrade\n")
                
                # 处理时间分析
                inference_time = stats['timing']['inference']['avg']
                preprocess_time = stats['timing']['preprocess']['avg']
                postprocess_time = stats['timing']['postprocess']['avg']
                total_time = inference_time + preprocess_time + postprocess_time
                
                f.write(f"\nProcessing Time Breakdown:\n")
                f.write(f"Preprocessing: {preprocess_time:.2f}ms ({preprocess_time/total_time*100:.1f}%)\n")
                f.write(f"Inference: {inference_time:.2f}ms ({inference_time/total_time*100:.1f}%)\n")
                f.write(f"Postprocessing: {postprocess_time:.2f}ms ({postprocess_time/total_time*100:.1f}%)\n")
                
                # 瓶颈分析
                f.write(f"\nBottleneck Analysis:\n")
                max_time = max(preprocess_time, inference_time, postprocess_time)
                if max_time == inference_time:
                    f.write("  → GPU/Model inference is the bottleneck\n")
                    f.write("    Suggestions: Consider smaller model or better GPU\n")
                elif max_time == preprocess_time:
                    f.write("  → Image preprocessing is the bottleneck\n")
                    f.write("    Suggestions: Optimize image processing or reduce input size\n")
                else:
                    f.write("  → Post-processing (NMS, etc.) is the bottleneck\n")
                    f.write("    Suggestions: Optimize NMS parameters or use GPU acceleration\n")
                
                # 资源利用分析
                cpu_avg = stats['resources']['cpu']['avg']
                memory_avg = stats['resources']['memory']['avg']
                
                f.write(f"\nResource Utilization Analysis:\n")
                f.write(f"CPU Usage: {cpu_avg:.1f}%\n")
                if cpu_avg > 300:
                    f.write("  → High multi-core utilization (normal for CPU-heavy tasks)\n")
                elif cpu_avg > 100:
                    f.write("  → Multi-core processing active\n")
                else:
                    f.write("  → Single-core or light processing\n")
                
                f.write(f"Memory Usage: {memory_avg:.1f}%\n")
                if memory_avg > 80:
                    f.write("  → High memory usage, monitor for potential issues\n")
                elif memory_avg > 50:
                    f.write("  → Moderate memory usage\n")
                else:
                    f.write("  → Low memory usage, system has capacity for larger models\n")
                
                # GPU分析
                if 'gpu' in stats['resources'] and stats['resources']['gpu']:
                    gpu_util = stats['resources']['gpu']['utilization']['avg']
                    gpu_mem = stats['resources']['gpu']['memory']['avg']
                    
                    f.write(f"GPU Utilization: {gpu_util:.1f}%\n")
                    if gpu_util > 80:
                        f.write("  → Excellent GPU utilization\n")
                    elif gpu_util > 50:
                        f.write("  → Good GPU utilization\n")
                    elif gpu_util > 20:
                        f.write("  → Moderate GPU utilization, consider larger batch size\n")
                    else:
                        f.write("  → Low GPU utilization, may be CPU-bound or small workload\n")
                    
                    f.write(f"GPU Memory: {gpu_mem:.1f}%\n")
                    if gpu_mem > 80:
                        f.write("  → High GPU memory usage, close to limit\n")
                    elif gpu_mem > 50:
                        f.write("  → Moderate GPU memory usage\n")
                    else:
                        f.write("  → Low GPU memory usage, can handle larger models/batches\n")
                else:
                    f.write("GPU: Not available or not utilized\n")
                    f.write("  → Consider using GPU acceleration for better performance\n")
                
                # 优化建议
                f.write(f"\nOptimization Recommendations:\n")
                f.write("-" * 30 + "\n")
                
                if fps < 15:
                    f.write("• Performance is below optimal:\n")
                    f.write("  - Try smaller model (yolov8n instead of yolov8s/m/l)\n")
                    f.write("  - Reduce input image size (--imgsz 416 or 320)\n")
                    f.write("  - Increase batch size if GPU memory allows\n")
                
                if cpu_avg > 400:
                    f.write("• High CPU usage detected:\n")
                    f.write("  - Consider reducing image preprocessing complexity\n")
                    f.write("  - Use GPU for preprocessing if available\n")
                
                if 'gpu' in stats['resources'] and stats['resources']['gpu']:
                    gpu_util = stats['resources']['gpu']['utilization']['avg']
                    if gpu_util < 50:
                        f.write("• Low GPU utilization:\n")
                        f.write("  - Increase batch size to better utilize GPU\n")
                        f.write("  - Check if preprocessing is creating bottleneck\n")
                
                # 稳定性分析
                inference_std = stats['timing']['inference']['std']
                f.write(f"\nPerformance Stability:\n")
                f.write(f"Inference time standard deviation: {inference_std:.2f}ms\n")
                if inference_std < 5:
                    f.write("  → Very stable performance\n")
                elif inference_std < 15:
                    f.write("  → Good stability\n")
                else:
                    f.write("  → Variable performance, check system load\n")
                
            print(f"   ✅ 性能摘要保存至: {summary_path}")
            
        except Exception as e:
            print(f"   ⚠️ 性能摘要创建失败: {e}")
    
    def save_results(self, stats):
        """保存结果到文件"""
        if not stats:
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        hostname = socket.gethostname()
        base_filename = f"{hostname}_{timestamp}"
        
        # 保存TXT格式
        if 'txt' in self.args.output_format:
            txt_filename = f"{base_filename}.txt"
            self.save_txt_report(stats, txt_filename)
            print(f"TXT报告保存至: {txt_filename}")
        
        # 保存JSON格式
        if 'json' in self.args.output_format:
            json_filename = f"{base_filename}.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"JSON报告保存至: {json_filename}")
        
        # 保存CSV格式
        if 'csv' in self.args.output_format:
            csv_filename = f"{base_filename}.csv"
            self.save_csv_report(stats, csv_filename)
            print(f"CSV报告保存至: {csv_filename}")
        
        # 生成可视化图表
        if self.args.plot:
            if VISUALIZATION_AVAILABLE:
                plot_filename = f"{base_filename}_visualization.png"
                try:
                    self.create_visualizations(stats, plot_filename)
                    print(f"📊 可视化图表保存至: {plot_filename}")
                except Exception as e:
                    print(f"❌ 生成图表失败: {e}")
                    print("请运行诊断工具检查环境")
            else:
                print("❌ 无法生成图表：matplotlib/seaborn不可用")
                print("安装方法: pip install matplotlib seaborn")
    
    def save_txt_report(self, stats, filename):
        """保存文本格式报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("===== YOLO性能基准测试报告 =====\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"主机名: {stats['system_info']['hostname']}\n")
            f.write(f"模型: {stats['system_info']['model']}\n")
            f.write(f"数据源: {stats['system_info']['source']}\n")
            f.write(f"设备: {stats['system_info']['device']}\n")
            f.write(f"批次大小: {stats['system_info']['batch_size']}\n")
            f.write(f"PyTorch版本: {stats['system_info']['torch_version']}\n")
            f.write(f"CUDA可用: {stats['system_info']['cuda_available']}\n")
            f.write(f"设备名称: {stats['system_info']['device_name']}\n\n")
            
            # 摘要
            f.write("===== 测试摘要 =====\n")
            f.write(f"总帧数: {stats['summary']['total_frames']}\n")
            f.write(f"总时间: {stats['summary']['total_time']:.2f} 秒\n")
            f.write(f"吞吐量: {stats['summary']['throughput']:.2f} FPS\n")
            f.write(f"平均帧时间: {stats['summary']['avg_frame_time']:.2f} 毫秒\n\n")
            
            # 时间详情
            f.write("===== 时间分解 =====\n")
            for stage, data in stats['timing'].items():
                stage_name = {'preprocess': '预处理', 'inference': '推理', 
                             'postprocess': '后处理', 'total_per_frame': '总计每帧'}
                f.write(f"{stage_name.get(stage, stage)}:\n")
                f.write(f"  最小值: {data['min']:.2f} 毫秒\n")
                f.write(f"  最大值: {data['max']:.2f} 毫秒\n")
                f.write(f"  平均值: {data['avg']:.2f} 毫秒\n")
                f.write(f"  标准差: {data['std']:.2f} 毫秒\n\n")
            
            # 资源使用
            f.write("===== 资源使用情况 =====\n")
            f.write(f"CPU使用率:\n")
            f.write(f"  最小值: {stats['resources']['cpu']['min']:.1f}%\n")
            f.write(f"  最大值: {stats['resources']['cpu']['max']:.1f}%\n")
            f.write(f"  平均值: {stats['resources']['cpu']['avg']:.1f}%\n\n")
            
            f.write(f"内存使用率:\n")
            f.write(f"  最小值: {stats['resources']['memory']['min']:.1f}%\n")
            f.write(f"  最大值: {stats['resources']['memory']['max']:.1f}%\n")
            f.write(f"  平均值: {stats['resources']['memory']['avg']:.1f}%\n\n")
            
            if 'gpu' in stats['resources']:
                f.write(f"GPU显存使用率:\n")
                f.write(f"  最小值: {stats['resources']['gpu']['memory']['min']:.1f}%\n")
                f.write(f"  最大值: {stats['resources']['gpu']['memory']['max']:.1f}%\n")
                f.write(f"  平均值: {stats['resources']['gpu']['memory']['avg']:.1f}%\n\n")
                
                f.write(f"GPU利用率:\n")
                f.write(f"  最小值: {stats['resources']['gpu']['utilization']['min']:.1f}%\n")
                f.write(f"  最大值: {stats['resources']['gpu']['utilization']['max']:.1f}%\n")
                f.write(f"  平均值: {stats['resources']['gpu']['utilization']['avg']:.1f}%\n\n")
    
    def save_csv_report(self, stats, filename):
        """保存CSV格式报告"""
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # 表头
            writer.writerow(['指标', '数值', '单位'])
            
            # 摘要数据
            writer.writerow(['总帧数', stats['summary']['total_frames'], '帧'])
            writer.writerow(['总时间', f"{stats['summary']['total_time']:.2f}", '秒'])
            writer.writerow(['吞吐量', f"{stats['summary']['throughput']:.2f}", 'FPS'])
            writer.writerow(['平均帧时间', f"{stats['summary']['avg_frame_time']:.2f}", '毫秒'])
            
            # 推理时间
            writer.writerow(['推理最小时间', f"{stats['timing']['inference']['min']:.2f}", '毫秒'])
            writer.writerow(['推理最大时间', f"{stats['timing']['inference']['max']:.2f}", '毫秒'])
            writer.writerow(['推理平均时间', f"{stats['timing']['inference']['avg']:.2f}", '毫秒'])
            
            # 资源使用
            writer.writerow(['CPU平均使用率', f"{stats['resources']['cpu']['avg']:.1f}", '%'])
            writer.writerow(['内存平均使用率', f"{stats['resources']['memory']['avg']:.1f}", '%'])
            
            if 'gpu' in stats['resources']:
                writer.writerow(['GPU平均利用率', f"{stats['resources']['gpu']['utilization']['avg']:.1f}", '%'])
                writer.writerow(['GPU平均显存使用', f"{stats['resources']['gpu']['memory']['avg']:.1f}", '%'])
    
    def print_summary(self, stats):
        """打印详细摘要到控制台"""
        if not stats:
            return
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Total frames processed: {stats['summary']['total_frames']}")
        print(f"Total time elapsed: {stats['summary']['total_time']:.2f} seconds")
        print(f"Throughput: {stats['summary']['throughput']:.2f} frames per second")
        
        # 详细时间统计
        print("\n" + "="*60)
        print("DETAILED METRICS")
        print("="*60)
        print("Preprocess time (ms):")
        print(f"  Min: {stats['timing']['preprocess']['min']:.2f}")
        print(f"  Max: {stats['timing']['preprocess']['max']:.2f}")
        print(f"  Avg: {stats['timing']['preprocess']['avg']:.2f}")
        
        print("\nInference time (ms):")
        print(f"  Min: {stats['timing']['inference']['min']:.2f}")
        print(f"  Max: {stats['timing']['inference']['max']:.2f}")
        print(f"  Avg: {stats['timing']['inference']['avg']:.2f}")
        
        print("\nPostprocess time (ms):")
        print(f"  Min: {stats['timing']['postprocess']['min']:.2f}")
        print(f"  Max: {stats['timing']['postprocess']['max']:.2f}")
        print(f"  Avg: {stats['timing']['postprocess']['avg']:.2f}")
        
        print("\nTotal processing time per frame (ms):")
        print(f"  Min: {stats['timing']['total_per_frame']['min']:.2f}")
        print(f"  Max: {stats['timing']['total_per_frame']['max']:.2f}")
        print(f"  Avg: {stats['timing']['total_per_frame']['avg']:.2f}")
        
        # 资源使用统计
        print("\n" + "="*60)
        print("RESOURCE UTILIZATION")
        print("="*60)
        
        # CPU使用率解释
        cpu_avg = stats['resources']['cpu']['avg']
        cpu_cores = stats['system_info']['cpu_cores']
        cpu_threads = stats['system_info']['cpu_threads']
        cpu_utilization_percent = (cpu_avg / (cpu_threads * 100)) * 100
        
        print("CPU Usage:")
        print(f"  Min: {stats['resources']['cpu']['min']:.2f}%")
        print(f"  Max: {stats['resources']['cpu']['max']:.2f}%")
        print(f"  Avg: {stats['resources']['cpu']['avg']:.2f}%")
        print(f"  CPU Info: {cpu_cores}核心/{cpu_threads}线程 (最大{cpu_threads * 100}%)")
        print(f"  CPU利用率: {cpu_utilization_percent:.1f}% 的总CPU能力")
        
        if cpu_avg > 100:
            print(f"  说明: 使用了{cpu_avg/100:.1f}个CPU核心，这是正常的多核并行")
        
        print("\nMemory Usage (%):")
        print(f"  Min: {stats['resources']['memory']['min']:.2f}")
        print(f"  Max: {stats['resources']['memory']['max']:.2f}")
        print(f"  Avg: {stats['resources']['memory']['avg']:.2f}")
        
        if 'gpu' in stats['resources']:
            print("\nGPU Memory Usage (%):")
            print(f"  Min: {stats['resources']['gpu']['memory']['min']:.2f}")
            print(f"  Max: {stats['resources']['gpu']['memory']['max']:.2f}")
            print(f"  Avg: {stats['resources']['gpu']['memory']['avg']:.2f}")
            
            print("\nGPU Utilization (%):")
            print(f"  Min: {stats['resources']['gpu']['utilization']['min']:.2f}")
            print(f"  Max: {stats['resources']['gpu']['utilization']['max']:.2f}")
            print(f"  Avg: {stats['resources']['gpu']['utilization']['avg']:.2f}")
            print(f"  Device: {stats['system_info']['device_name']}")
        else:
            print("\nGPU: Not available")
        
        # 性能评级
        fps = stats['summary']['throughput']
        if fps > 30:
            rating = "Excellent 🟢"
        elif fps > 15:
            rating = "Good 🟡"
        elif fps > 5:
            rating = "Fair 🟠"
        else:
            rating = "Slow 🔴"
        
        print(f"\nPerformance Rating: {rating}")
        print(f"Model: {stats['system_info']['model']}")
        print(f"Device: {stats['system_info']['device']}")
        print(f"Batch Size: {stats['system_info']['batch_size']}")


def main():
    """主函数"""
    print("YOLOv8 性能基准测试工具 - 改进版")
    print("-" * 50)
    
    # 解析参数
    args = parse_arguments()
    
    # 创建基准测试工具
    benchmark = BenchmarkTool(args)
    
    # 加载模型
    benchmark.load_model()
    
    # 运行测试
    success = benchmark.run_benchmark()
    
    if success:
        # 计算统计数据
        stats = benchmark.calculate_statistics()
        
        # 打印摘要
        benchmark.print_summary(stats)
        
        # 保存结果
        benchmark.save_results(stats)
        
        print("\n测试完成!")
    else:
        print("测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main()