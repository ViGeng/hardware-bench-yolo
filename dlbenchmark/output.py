#!/usr/bin/env python3
"""
输出模块 - 负责结果保存和可视化生成
"""

import os
import csv
import time
import socket
import logging
import numpy as np
from utils import check_dependencies

# 检查依赖
dependencies = check_dependencies()

class ResultExporter:
    """结果导出器"""
    
    def __init__(self, detailed_results=None, results_dir=None):
        self.detailed_results = detailed_results or []
        self.logger = logging.getLogger(__name__)
        
        # 使用指定的目录或默认的results目录
        self.results_dir = results_dir or "results"
        
        # 确保输出目录存在
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            self.logger.info(f"创建输出目录: {self.results_dir}")
    
    def save_detailed_csv_results(self, stats, model_type):
        """保存详细的CSV结果"""
        self.logger.info(f"开始保存详细CSV结果到目录: {self.results_dir}")
        
        # 生成带时间戳的文件名
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 详细结果文件
        detailed_filename = os.path.join(self.results_dir, f"{model_type}_detailed_{timestamp}.csv")
        
        try:
            with open(detailed_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 写入表头
                if model_type == 'detection':
                    header = ['Image_ID', 'Preprocessing_Time_ms', 'Inference_Time_ms', 'Postprocessing_Time_ms', 'Rendering_Time_ms', 'Total_Time_ms']
                elif model_type == 'segmentation':
                    header = ['Sample_ID', 'Preprocessing_Time_ms', 'Inference_Time_ms', 'Postprocessing_Time_ms', 'Rendering_Time_ms', 'Total_Time_ms']
                else:
                    header = ['Sample_ID', 'Preprocessing_Time_ms', 'Inference_Time_ms', 'Postprocessing_Time_ms', 'Rendering_Time_ms', 'Total_Time_ms']
                
                writer.writerow(header)
                
                # 写入详细数据
                for result in self.detailed_results:
                    writer.writerow([
                        result['sample_id'],
                        f"{result['preprocessing_time']:.4f}",
                        f"{result['inference_time']:.4f}",
                        f"{result['postprocessing_time']:.4f}",
                        f"{result['rendering_time']:.4f}",
                        f"{result['total_time']:.4f}"
                    ])
            
            self.logger.info(f"详细结果已保存至: {detailed_filename}")
            print(f"详细结果已保存至: {detailed_filename}")
            
        except Exception as e:
            self.logger.error(f"保存详细结果文件失败: {e}")
            raise e
        
        # 汇总统计文件
        summary_filename = os.path.join(self.results_dir, f"{model_type}_summary_{timestamp}.csv")
        
        try:
            with open(summary_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 系统信息部分
                writer.writerow(['=== SYSTEM INFORMATION ==='])
                writer.writerow(['Metric', 'Value', 'Unit'])
                writer.writerow(['Hostname', stats['system_info']['hostname'], ''])
                writer.writerow(['Model Type', stats['system_info']['model_type'], ''])
                writer.writerow(['Model Name', stats['system_info']['model_name'], ''])
                writer.writerow(['Dataset', stats['system_info']['dataset'], ''])
                writer.writerow(['Device', stats['system_info']['device'], ''])
                writer.writerow(['Device Name', stats['system_info']['device_name'], ''])
                writer.writerow(['PyTorch Version', stats['system_info']['torch_version'], ''])
                writer.writerow(['CUDA Available', stats['system_info']['cuda_available'], ''])
                writer.writerow(['Test Start Time', time.strftime('%Y-%m-%d %H:%M:%S'), ''])
                writer.writerow([])
                
                # 性能指标部分
                writer.writerow(['=== PERFORMANCE METRICS ==='])
                writer.writerow(['Metric', 'Value', 'Unit'])
                writer.writerow(['Total Samples', stats['performance']['total_samples'], 'samples'])
                writer.writerow(['Total Time', f"{stats['performance']['total_time']:.4f}", 'seconds'])
                writer.writerow(['Throughput', f"{stats['performance']['throughput']:.4f}", 'samples/sec'])
                writer.writerow(['Avg Time per Sample', f"{stats['performance']['avg_time_per_sample']:.4f}", 'ms'])
                if 'rating' in stats['performance']:
                    writer.writerow(['Performance Rating', stats['performance']['rating'], ''])
                writer.writerow([])
                
                # 时间分解部分
                if stats['timing']:
                    writer.writerow(['=== TIMING BREAKDOWN ==='])
                    writer.writerow(['Stage', 'Min (ms)', 'Max (ms)', 'Avg (ms)', 'Std (ms)'])
                    for stage, data in stats['timing'].items():
                        stage_name = stage.replace('_', ' ').title()
                        writer.writerow([
                            stage_name,
                            f"{data['min']:.4f}",
                            f"{data['max']:.4f}",
                            f"{data['avg']:.4f}",
                            f"{data['std']:.4f}"
                        ])
                    writer.writerow([])
                
                # 资源使用部分
                writer.writerow(['=== RESOURCE UTILIZATION ==='])
                writer.writerow(['Resource', 'Min (%)', 'Max (%)', 'Avg (%)', 'Std (%)'])
                writer.writerow(['CPU Usage', 
                               f"{stats['resources']['cpu']['min']:.2f}", 
                               f"{stats['resources']['cpu']['max']:.2f}", 
                               f"{stats['resources']['cpu']['avg']:.2f}",
                               f"{stats['resources']['cpu']['std']:.2f}"])
                writer.writerow(['Memory Usage', 
                               f"{stats['resources']['memory']['min']:.2f}", 
                               f"{stats['resources']['memory']['max']:.2f}", 
                               f"{stats['resources']['memory']['avg']:.2f}",
                               f"{stats['resources']['memory']['std']:.2f}"])
                
                if 'gpu' in stats['resources']:
                    writer.writerow(['GPU Memory', 
                                   f"{stats['resources']['gpu']['memory']['min']:.2f}", 
                                   f"{stats['resources']['gpu']['memory']['max']:.2f}", 
                                   f"{stats['resources']['gpu']['memory']['avg']:.2f}",
                                   f"{stats['resources']['gpu']['memory']['std']:.2f}"])
                    writer.writerow(['GPU Utilization', 
                                   f"{stats['resources']['gpu']['utilization']['min']:.2f}", 
                                   f"{stats['resources']['gpu']['utilization']['max']:.2f}", 
                                   f"{stats['resources']['gpu']['utilization']['avg']:.2f}",
                                   f"{stats['resources']['gpu']['utilization']['std']:.2f}"])
            
            self.logger.info(f"汇总结果已保存至: {summary_filename}")
            print(f"汇总结果已保存至: {summary_filename}")
            
        except Exception as e:
            self.logger.error(f"保存汇总结果文件失败: {e}")
            raise e
        
        return [detailed_filename, summary_filename]

class Visualizer:
    """可视化生成器"""
    
    def __init__(self, detailed_results=None, results_dir=None):
        self.detailed_results = detailed_results or []
        self.logger = logging.getLogger(__name__)
        self.matplotlib_available = dependencies['matplotlib']
        
        # 使用指定的目录或默认的results目录
        self.results_dir = results_dir or "results"
        
        # 确保输出目录存在
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            self.logger.info(f"创建可视化输出目录: {self.results_dir}")
    
    def _calculate_moving_average(self, data, window_size):
        """计算滑动窗口平均值"""
        if len(data) < window_size:
            # 如果数据量小于窗口大小，返回累积平均
            return [np.mean(data[:i+1]) for i in range(len(data))]
        
        moving_avg = []
        for i in range(len(data)):
            if i < window_size - 1:
                # 前面不足窗口大小的部分，使用从开始到当前位置的平均值
                moving_avg.append(np.mean(data[:i+1]))
            else:
                # 使用滑动窗口平均
                window_data = data[i-window_size+1:i+1]
                moving_avg.append(np.mean(window_data))
        
        return moving_avg
    
    def create_visualizations(self, stats, model_type):
        """创建可视化图表"""
        if not self.matplotlib_available:
            self.logger.warning("matplotlib不可用，跳过可视化生成")
            print("matplotlib不可用，跳过可视化生成")
            return []
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            
            self.logger.info(f"开始生成可视化图表到目录: {self.results_dir}")
            print("正在生成可视化图表...")
            
            generated_plots = []
            
            # 生成详细时间折线图
            timing_plot = self._create_detailed_timing_plot(stats, model_type)
            if timing_plot:
                generated_plots.append(timing_plot)
            
            # 生成性能总结图表
            summary_plot = self._create_summary_plot(stats, model_type)
            if summary_plot:
                generated_plots.append(summary_plot)
            
            self.logger.info(f"可视化图表生成完成，共生成 {len(generated_plots)} 个图表文件")
            
            return generated_plots
            
        except Exception as e:
            self.logger.error(f"创建可视化时出错: {e}")
            print(f"创建可视化时出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_detailed_timing_plot(self, stats, model_type):
        """创建详细的每帧速度分析折线图"""
        if not self.detailed_results or len(self.detailed_results) < 10:
            self.logger.warning("数据不足，跳过详细时间折线图生成")
            print("数据不足，跳过详细时间折线图生成")
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            self.logger.info("开始生成详细速度分析折线图")
            print("正在生成详细速度分析折线图...")
            
            # 提取数据
            sample_ids = [r['sample_id'] for r in self.detailed_results]
            total_times = [max(r['total_time'], 0.001) for r in self.detailed_results]
            inf_times = [max(r['inference_time'], 0.001) for r in self.detailed_results]
            prep_times = [r['preprocessing_time'] for r in self.detailed_results]
            post_times = [r['postprocessing_time'] for r in self.detailed_results]
            render_times = [r['rendering_time'] for r in self.detailed_results]
            
            # 计算FPS
            fps_total = [min(1000.0 / t, 10000) for t in total_times]
            fps_inference = [min(1000.0 / t, 10000) for t in inf_times]
            
            # 计算全局平均值
            avg_fps_total = np.mean(fps_total)
            avg_fps_inference = np.mean(fps_inference)
            
            # 设置滑动窗口大小
            data_length = len(fps_total)
            if data_length >= 100:
                window_size = 20  # 大数据集使用20个样本的窗口
            elif data_length >= 50:
                window_size = 10  # 中等数据集使用10个样本的窗口
            else:
                window_size = max(3, data_length // 5)  # 小数据集使用1/5数据量作为窗口，最少3个
            
            self.logger.info(f"使用滑动窗口大小: {window_size} (数据总量: {data_length})")
            
            # 计算滑动窗口平均
            fps_total_ma = self._calculate_moving_average(fps_total, window_size)
            fps_inference_ma = self._calculate_moving_average(fps_inference, window_size)
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            model_name = stats["system_info"]["model_name"]
            dataset_name = stats["system_info"]["dataset"]
            fig.suptitle(f'Per-Frame Speed Analysis: {model_name} on {dataset_name}', fontsize=16)
            
            # 上图：FPS性能图
            # 原始FPS数据（半透明）
            ax1.plot(sample_ids, fps_total, label='Total FPS (Raw)', color='lightblue', alpha=0.5, linewidth=1)
            ax1.plot(sample_ids, fps_inference, label='Inference FPS (Raw)', color='lightcoral', alpha=0.5, linewidth=1)
            
            # 滑动窗口平均FPS（醒目）
            ax1.plot(sample_ids, fps_total_ma, label=f'Total FPS (MA-{window_size})', color='blue', alpha=0.8, linewidth=2)
            ax1.plot(sample_ids, fps_inference_ma, label=f'Inference FPS (MA-{window_size})', color='red', alpha=0.8, linewidth=2)
            
            # 添加全局平均值线
            ax1.axhline(y=avg_fps_total, color='blue', linestyle='--', alpha=0.6, linewidth=1.5, 
                       label=f'Global Avg Total FPS: {avg_fps_total:.1f}')
            ax1.axhline(y=avg_fps_inference, color='red', linestyle='--', alpha=0.6, linewidth=1.5,
                       label=f'Global Avg Inference FPS: {avg_fps_inference:.1f}')
            
            # 添加当前滑动窗口平均值（最后几个值的平均）
            current_total_ma = np.mean(fps_total_ma[-window_size:]) if len(fps_total_ma) >= window_size else fps_total_ma[-1]
            current_inf_ma = np.mean(fps_inference_ma[-window_size:]) if len(fps_inference_ma) >= window_size else fps_inference_ma[-1]
            
            ax1.axhline(y=current_total_ma, color='darkblue', linestyle=':', alpha=0.8, linewidth=2,
                       label=f'Current MA Total FPS: {current_total_ma:.1f}')
            ax1.axhline(y=current_inf_ma, color='darkred', linestyle=':', alpha=0.8, linewidth=2,
                       label=f'Current MA Inference FPS: {current_inf_ma:.1f}')
            
            ax1.set_xlabel('Sample/Image ID')
            ax1.set_ylabel('FPS (Frames Per Second)')
            ax1.set_title(f'Processing Speed per Frame (Moving Average Window: {window_size})')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 下图：时间分解堆积图
            ax2.fill_between(sample_ids, 0, prep_times, label='Preprocessing', alpha=0.8, color='lightblue')
            current_height = prep_times
            
            inf_height = [p + i for p, i in zip(current_height, inf_times)]
            ax2.fill_between(sample_ids, current_height, inf_height, label='Inference', alpha=0.8, color='lightcoral')
            current_height = inf_height
            
            post_height = [p + post for p, post in zip(current_height, post_times)]
            ax2.fill_between(sample_ids, current_height, post_height, label='Postprocessing', alpha=0.8, color='lightgreen')
            current_height = post_height
            
            render_height = [p + r for p, r in zip(current_height, render_times)]
            ax2.fill_between(sample_ids, current_height, render_height, label='Rendering', alpha=0.8, color='gold')
            
            ax2.set_xlabel('Sample/Image ID')
            ax2.set_ylabel('Time (ms)')
            ax2.set_title('Time Breakdown per Sample')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            timing_plot_filename = os.path.join(self.results_dir, f"{model_type}_speed_analysis_{timestamp}.png")
            plt.savefig(timing_plot_filename, format='png', dpi=300, bbox_inches='tight')
            
            self.logger.info(f"详细速度分析图表已保存至: {timing_plot_filename}")
            print(f"详细速度分析图表已保存至: {timing_plot_filename}")
            
            plt.close(fig)
            return timing_plot_filename
            
        except Exception as e:
            self.logger.error(f"创建详细速度折线图时出错: {e}")
            print(f"创建详细速度折线图时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_summary_plot(self, stats, model_type):
        """创建性能总结图表"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            model_name = stats["system_info"]["model_name"]
            dataset_name = stats["system_info"]["dataset"]
            fig.suptitle(f'Benchmark Results Summary: {model_name} on {dataset_name}', fontsize=16)
            
            # 1. 时间分解饼图
            if stats['timing']:
                timing_data = [(k.replace('_', ' ').title(), v['avg']) for k, v in stats['timing'].items()]
                labels, values = zip(*timing_data)
                ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Timing Breakdown')
            else:
                ax1.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Timing Breakdown')
            
            # 2. 资源利用率柱状图
            resources = ['CPU', 'Memory']
            usage = [stats['resources']['cpu']['avg'], stats['resources']['memory']['avg']]
            errors = [stats['resources']['cpu']['std'], stats['resources']['memory']['std']]
            
            if 'gpu' in stats['resources']:
                resources.extend(['GPU Memory', 'GPU Util'])
                usage.extend([stats['resources']['gpu']['memory']['avg'], 
                             stats['resources']['gpu']['utilization']['avg']])
                errors.extend([stats['resources']['gpu']['memory']['std'],
                              stats['resources']['gpu']['utilization']['std']])
            
            colors = ['skyblue', 'lightgreen', 'orange', 'red']
            bars = ax2.bar(resources, usage, yerr=errors, capsize=5, color=colors[:len(resources)])
            ax2.set_title('Resource Utilization')
            ax2.set_ylabel('Usage (%)')
            ax2.set_ylim(0, 100)
            
            # 3. 性能时间分布
            if self.detailed_results and len(self.detailed_results) > 10:
                total_times = [r['total_time'] for r in self.detailed_results]
                ax3.hist(total_times, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
                ax3.set_xlabel('Total Time per Sample (ms)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Processing Time Distribution')
                mean_time = np.mean(total_times)
                ax3.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.2f}ms')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'Insufficient data\nfor distribution plot', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Processing Time Distribution')
            
            # 4. 系统信息和性能总结
            device_info = stats['system_info']['device']
            model_info = stats['system_info']['model_name']
            dataset_info = stats['system_info']['dataset']
            throughput = stats['performance']['throughput']
            avg_time = stats['performance']['avg_time_per_sample']
            total_samples = stats['performance']['total_samples']
            
            cpu_avg = stats['resources']['cpu']['avg']
            cpu_std = stats['resources']['cpu']['std']
            mem_avg = stats['resources']['memory']['avg']
            mem_std = stats['resources']['memory']['std']
            
            if 'gpu' in stats['resources']:
                gpu_mem_avg = stats['resources']['gpu']['memory']['avg']
                gpu_mem_std = stats['resources']['gpu']['memory']['std']
                gpu_util_avg = stats['resources']['gpu']['utilization']['avg']
                gpu_util_std = stats['resources']['gpu']['utilization']['std']
                
                system_text = f"""System Information:
Device: {device_info}
Model: {model_info}
Dataset: {dataset_info}

Performance Summary:
Throughput: {throughput:.2f} samples/sec
Avg time/sample: {avg_time:.2f}ms
Total samples: {total_samples}

Resource Usage (Avg ± Std):
CPU: {cpu_avg:.1f}% ± {cpu_std:.1f}%
Memory: {mem_avg:.1f}% ± {mem_std:.1f}%
GPU Mem: {gpu_mem_avg:.1f}% ± {gpu_mem_std:.1f}%
GPU Util: {gpu_util_avg:.1f}% ± {gpu_util_std:.1f}%"""
            else:
                system_text = f"""System Information:
Device: {device_info}
Model: {model_info}
Dataset: {dataset_info}

Performance Summary:
Throughput: {throughput:.2f} samples/sec
Avg time/sample: {avg_time:.2f}ms
Total samples: {total_samples}

Resource Usage (Avg ± Std):
CPU: {cpu_avg:.1f}% ± {cpu_std:.1f}%
Memory: {mem_avg:.1f}% ± {mem_std:.1f}%"""
            
            ax4.text(0.05, 0.95, system_text, transform=ax4.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('System Info & Performance Summary')
            
            plt.tight_layout()
            
            # 保存总结图表
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            summary_plot_filename = os.path.join(self.results_dir, f"{model_type}_summary_{timestamp}.png")
            plt.savefig(summary_plot_filename, format='png', dpi=300, bbox_inches='tight')
            
            self.logger.info(f"性能总结图表已保存至: {summary_plot_filename}")
            print(f"性能总结图表已保存至: {summary_plot_filename}")
            
            plt.close(fig)
            return summary_plot_filename
            
        except Exception as e:
            self.logger.error(f"创建总结图表时出错: {e}")
            print(f"创建总结图表时出错: {e}")
            import traceback
            traceback.print_exc()
            return None