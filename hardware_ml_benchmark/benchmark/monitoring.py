#!/usr/bin/env python3
"""
Monitoring module - responsible for system resource monitoring and statistics calculation, including monitoring overhead analysis
"""

import time
import threading
import logging
import socket
from collections import deque
import numpy as np
import torch
import psutil
from hardware_ml_benchmark.core.utils import get_system_info, check_dependencies

# Check dependencies
dependencies = check_dependencies()

class MonitoringOverheadAnalyzer:
    """Monitoring overhead analyzer"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.overhead_samples = []
        self.baseline_samples = []
        self.monitoring_active = False
        
    def measure_monitoring_overhead(self, duration=5.0, sample_interval=0.1):
        """Measure the overhead of the monitoring system itself"""
        self.logger.info(f"Starting to measure monitoring overhead, test duration: {duration} seconds")
        print(f"\n📊 Measuring monitoring system overhead...")
        print(f"Test duration: {duration} seconds, sampling interval: {sample_interval} seconds")
        
        # 1. Measure baseline performance without monitoring
        print("🔍 Measuring baseline performance (no monitoring)...")
        self.baseline_samples = self._run_dummy_workload(duration, with_monitoring=False)
        
        # 2. Measure performance with monitoring
        print("🔍 Measuring monitoring overhead (with monitoring)...")
        self.overhead_samples = self._run_dummy_workload(duration, with_monitoring=True)
        
        # 3. Calculate overhead statistics
        overhead_stats = self._calculate_overhead_stats()
        
        return overhead_stats
    
    def _run_dummy_workload(self, duration, with_monitoring=False):
        """Run dummy workload to measure performance"""
        samples = []
        monitor = None
        
        if with_monitoring:
            # Start monitoring
            monitor = ResourceMonitor(enable_overhead_measurement=True)
            monitor.start_monitoring()
            time.sleep(0.1)  # Let monitoring stabilize
        
        start_time = time.time()
        iteration_count = 0
        
        while time.time() - start_time < duration:
            # Simulate typical deep learning operations
            iteration_start = time.time()
            
            # Simulate data preprocessing
            data = torch.randn(1, 3, 224, 224)
            if torch.cuda.is_available():
                data = data.cuda()
            
            # Simulate simple computation
            result = torch.relu(data)
            result = torch.mean(result)
            
            # Simulate data transfer
            if torch.cuda.is_available():
                result = result.cpu()
            
            iteration_time = (time.time() - iteration_start) * 1000  # ms
            samples.append(iteration_time)
            iteration_count += 1
        
        if with_monitoring and monitor:
            monitor.stop_monitoring()
        
        total_time = time.time() - start_time
        self.logger.info(f"Workload completed: {iteration_count} iterations, total duration: {total_time:.2f} seconds")
        
        return samples
    
    def _calculate_overhead_stats(self):
        """Calculate monitoring overhead statistics"""
        if not self.baseline_samples or not self.overhead_samples:
            return None
        
        baseline_mean = np.mean(self.baseline_samples)
        overhead_mean = np.mean(self.overhead_samples)
        
        baseline_std = np.std(self.baseline_samples)
        overhead_std = np.std(self.overhead_samples)
        
        # Calculate overhead
        absolute_overhead = overhead_mean - baseline_mean
        relative_overhead = (absolute_overhead / baseline_mean) * 100
        
        # Calculate throughput impact
        baseline_fps = 1000.0 / baseline_mean
        overhead_fps = 1000.0 / overhead_mean
        fps_impact = ((baseline_fps - overhead_fps) / baseline_fps) * 100
        
        stats = {
            'baseline': {
                'mean_time_ms': baseline_mean,
                'std_time_ms': baseline_std,
                'fps': baseline_fps,
                'samples': len(self.baseline_samples)
            },
            'with_monitoring': {
                'mean_time_ms': overhead_mean,
                'std_time_ms': overhead_std,
                'fps': overhead_fps,
                'samples': len(self.overhead_samples)
            },
            'overhead': {
                'absolute_ms': absolute_overhead,
                'relative_percent': relative_overhead,
                'fps_impact_percent': fps_impact
            },
            'accuracy_rating': self._get_accuracy_rating(relative_overhead)
        }
        
        return stats
    
    def _get_accuracy_rating(self, relative_overhead):
        """Get accuracy rating based on relative overhead"""
        if relative_overhead < 1.0:
            return "🟢 Excellent (< 1% overhead)"
        elif relative_overhead < 3.0:
            return "🟡 Good (< 3% overhead)"
        elif relative_overhead < 5.0:
            return "🟠 Fair (< 5% overhead)"
        else:
            return "🔴 Poor (≥ 5% overhead)"
    
    def print_overhead_analysis(self, stats):
        """Print monitoring overhead analysis results"""
        if not stats:
            print("❌ Unable to generate overhead analysis report")
            return
        
        print("\n" + "="*70)
        print("📊 MONITORING OVERHEAD ANALYSIS")
        print("="*70)
        
        print(f"🔧 Baseline Performance (No Monitoring):")
        print(f"  Mean time per operation: {stats['baseline']['mean_time_ms']:.3f}ms")
        print(f"  Standard deviation: {stats['baseline']['std_time_ms']:.3f}ms")
        print(f"  Throughput: {stats['baseline']['fps']:.1f} FPS")
        print(f"  Sample count: {stats['baseline']['samples']}")
        
        print(f"\n📈 Performance with Monitoring:")
        print(f"  Mean time per operation: {stats['with_monitoring']['mean_time_ms']:.3f}ms")
        print(f"  Standard deviation: {stats['with_monitoring']['std_time_ms']:.3f}ms")
        print(f"  Throughput: {stats['with_monitoring']['fps']:.1f} FPS")
        print(f"  Sample count: {stats['with_monitoring']['samples']}")
        
        print(f"\n⚡ Monitoring Overhead:")
        print(f"  Absolute overhead: {stats['overhead']['absolute_ms']:.3f}ms per operation")
        print(f"  Relative overhead: {stats['overhead']['relative_percent']:.2f}%")
        print(f"  FPS impact: {stats['overhead']['fps_impact_percent']:.2f}%")
        
        print(f"\n🎯 Monitoring Accuracy Rating:")
        print(f"  {stats['accuracy_rating']}")
        
        # Log final results
        self.logger.info(f"🎯 Monitoring Accuracy Rating: {stats['accuracy_rating']}")
        
        # Recommendations
        if stats['overhead']['relative_percent'] > 5.0:
            print(f"\n💡 Recommendations:")
            print(f"  - Consider increasing monitoring sampling interval")
            print(f"  - Disable detailed GPU monitoring")
            print(f"  - Reduce monitoring data retention count")
        elif stats['overhead']['relative_percent'] < 1.0:
            advice_msg = "✅ Monitoring overhead is minimal, impact on test results is negligible"
            print(f"\n{advice_msg}")
            self.logger.info(advice_msg)
        
        print("="*70)

class ResourceMonitor:
    """System resource monitor (optimized version)"""
    
    def __init__(self, enable_gpu_monitoring=True, sample_interval=0.1, max_samples=1000, enable_overhead_measurement=False):
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.sample_interval = sample_interval
        self.max_samples = max_samples
        self.enable_overhead_measurement = enable_overhead_measurement
        
        # Monitoring data storage
        self.cpu_usage = deque(maxlen=max_samples)
        self.memory_usage = deque(maxlen=max_samples)
        self.gpu_memory = deque(maxlen=max_samples)
        self.gpu_utilization = deque(maxlen=max_samples)
        
        # Monitoring overhead measurement
        self.monitoring_overhead_times = deque(maxlen=1000) if enable_overhead_measurement else None
        
        # GPU monitoring related
        self.nvml_available = dependencies['pynvml'] and enable_gpu_monitoring
        self.gpu_handle = None
        
        if torch.cuda.is_available() and self.nvml_available:
            self._try_initialize_gpu_monitoring()
        elif not enable_gpu_monitoring:
            pass  # Silent, don't log
        elif not torch.cuda.is_available():
            pass  # Silent, don't log  
        elif not dependencies['pynvml']:
            pass  # Silent, don't log
    
    def _try_initialize_gpu_monitoring(self):
        """Try to initialize GPU monitoring, including PATH fixes"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # Only log on success
            if not self.enable_overhead_measurement:
                self.logger.info("Detailed GPU monitoring available")
            return
        except pynvml.NVMLError_LibraryNotFound:
            # Silently try to fix PATH issues
            if self._fix_nvidia_path():
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    if not self.enable_overhead_measurement:
                        self.logger.info("Detailed GPU monitoring available")
                    return
                except Exception:
                    pass  # Silent failure
            
            # Don't show detailed failure info, GPU monitoring is not core functionality
            self.gpu_handle = None
        except Exception:
            # Handle other exceptions silently too
            self.gpu_handle = None
    
    def _fix_nvidia_path(self):
        """Try to fix NVIDIA path issues"""
        import os
        import glob
        
        # Common NVIDIA paths on Windows
        base_nvidia_paths = [
            r"C:\Program Files\NVIDIA Corporation\NVSMI",
            r"C:\Windows\System32",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.*\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.*\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.*\bin",
        ]
        
        # Expand wildcard paths
        nvidia_paths = []
        for base_path in base_nvidia_paths:
            if '*' in base_path:
                nvidia_paths.extend(glob.glob(base_path))
            else:
                nvidia_paths.append(base_path)
        
        # Silent search for nvidia-smi.exe
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
        
        # Only return True if paths were added, don't log details
        return len(added_paths) > 0
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if not self.enable_overhead_measurement:
            self.logger.info(f"Starting system resource monitoring (sampling interval: {self.sample_interval}s)")
        self.monitoring = True
        
        monitor_thread = threading.Thread(target=self._monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return monitor_thread
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        time.sleep(self.sample_interval * 2)  # Wait for monitoring thread to end
        if not self.enable_overhead_measurement:
            self.logger.info("System resource monitoring ended")
    
    def _monitor_resources(self):
        """Monitor system resources"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                monitor_start = time.time()
                
                # CPU and memory
                self.cpu_usage.append(process.cpu_percent(interval=0))
                self.memory_usage.append(psutil.virtual_memory().percent)
                
                # GPU monitoring
                if torch.cuda.is_available():
                    # GPU memory
                    gpu_mem = torch.cuda.memory_allocated(0)
                    gpu_total = torch.cuda.get_device_properties(0).total_memory
                    self.gpu_memory.append((gpu_mem / gpu_total) * 100)
                    
                    # GPU utilization
                    if self.gpu_handle and self.nvml_available:
                        try:
                            import pynvml
                            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                            self.gpu_utilization.append(util.gpu)
                        except:
                            self.gpu_utilization.append(0)
                
                # Record monitoring overhead (if enabled)
                if self.monitoring_overhead_times is not None:
                    monitor_time = (time.time() - monitor_start) * 1000  # ms
                    self.monitoring_overhead_times.append(monitor_time)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                if not self.enable_overhead_measurement:
                    self.logger.error(f"Error during resource monitoring: {e}")
                break
    
    def get_resource_stats(self):
        """Get resource usage statistics"""
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
        
        # GPU statistics
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
        
        # Monitoring overhead statistics
        if self.monitoring_overhead_times:
            stats['monitoring_overhead'] = {
                'avg_time_ms': np.mean(self.monitoring_overhead_times),
                'max_time_ms': np.max(self.monitoring_overhead_times),
                'samples': len(self.monitoring_overhead_times)
            }
        
        return stats

class StatisticsCalculator:
    """Statistics calculator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_benchmark_statistics(self, timing_results, total_time, total_samples, 
                                     model_type, model_info, dataset_name, device, resource_stats):
        """Calculate benchmark test statistics"""
        self.logger.info("Starting to calculate statistics")
        
        # Get system information
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
        
        # Add timing statistics
        for key, times in timing_results.items():
            if times:
                stats['timing'][key] = {
                    'min': np.min(times),
                    'max': np.max(times),
                    'avg': np.mean(times),
                    'std': np.std(times)
                }
        
        # Add monitoring overhead information (if available)
        if 'monitoring_overhead' in resource_stats:
            overhead_info = resource_stats['monitoring_overhead']
            stats['monitoring_accuracy'] = {
                'avg_overhead_ms': overhead_info['avg_time_ms'],
                'max_overhead_ms': overhead_info['max_time_ms'],
                'overhead_samples': overhead_info['samples']
            }
        
        # Log key statistics to log file
        fps = stats['performance']['throughput']
        self.logger.info(f"Statistics calculation completed - throughput: {stats['performance']['throughput']:.2f} samples/sec")
        self.logger.info(f"FPS: {fps:.2f}")
        
        return stats
    
    def print_results_summary(self, stats):
        """Print concise results summary"""
        self.logger.info("Starting to print test results")
        
        print("\n" + "="*70)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*70)
        
        # Basic information
        print(f"Model: {stats['system_info']['model_name']}")
        print(f"Dataset: {stats['system_info']['dataset']}")
        print(f"Device: {stats['system_info']['device']}")
        print(f"Device Name: {stats['system_info']['device_name']}")
        
        # Performance metrics
        print(f"\n{'='*20} PERFORMANCE METRICS {'='*20}")
        print(f"  Samples processed: {stats['performance']['total_samples']}")
        print(f"  Total time: {stats['performance']['total_time']:.2f}s")
        print(f"  Throughput: {stats['performance']['throughput']:.2f} samples/sec")
        print(f"  Avg time/sample: {stats['performance']['avg_time_per_sample']:.2f}ms")
        
        # Timing breakdown
        if stats['timing']:
            print(f"\n{'='*20} TIMING BREAKDOWN (ms) {'='*20}")
            for stage, data in stats['timing'].items():
                stage_name = stage.replace('_', ' ').title()
                print(f"  {stage_name}:")
                print(f"    Min: {data['min']:.2f}ms")
                print(f"    Max: {data['max']:.2f}ms")
                print(f"    Avg: {data['avg']:.2f}ms ± {data['std']:.2f}")
                print()
        
        # Resource usage
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
        
        # Monitoring accuracy information
        if 'monitoring_accuracy' in stats:
            print(f"{'='*20} MONITORING ACCURACY {'='*20}")
            print(f"  Avg monitoring overhead: {stats['monitoring_accuracy']['avg_overhead_ms']:.3f}ms per sample")
            print(f"  Max monitoring overhead: {stats['monitoring_accuracy']['max_overhead_ms']:.3f}ms per sample")
            print(f"  Monitoring samples: {stats['monitoring_accuracy']['overhead_samples']}")
            
            # Calculate relative overhead
            if stats['performance']['avg_time_per_sample'] > 0:
                relative_overhead = (stats['monitoring_accuracy']['avg_overhead_ms'] / stats['performance']['avg_time_per_sample']) * 100
                print(f"  Relative overhead: {relative_overhead:.2f}%")
                
                if relative_overhead < 1.0:
                    print(f"  🟢 Monitoring accuracy: Excellent (< 1% overhead)")
                elif relative_overhead < 3.0:
                    print(f"  🟡 Monitoring accuracy: Good (< 3% overhead)")
                elif relative_overhead < 5.0:
                    print(f"  🟠 Monitoring accuracy: Fair (< 5% overhead)")
                else:
                    print(f"  🔴 Monitoring accuracy: Poor (≥ 5% overhead)")
        
        print("="*70)