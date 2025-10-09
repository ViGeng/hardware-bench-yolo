#!/usr/bin/env python3
"""
Deep Learning Benchmark Tool
Main program entry module

Features:
1. Command line argument parsing and validation
2. Model loading and benchmarking
3. Result saving and visualization
4. Resource monitoring

Command line usage examples:
python -m hardware_ml_benchmark.main --device cuda:0 --task classification --model resnet18 --dataset MNIST --samples 100
python -m hardware_ml_benchmark.main --device cpu --task detection --model yolov8n --dataset Test-Images --samples 500
python -m hardware_ml_benchmark.main --list-models
python -m hardware_ml_benchmark.main --list-datasets
"""

import logging
import os
import sys
import time

from hardware_ml_benchmark.benchmark.benchmarks import BenchmarkRunner
from hardware_ml_benchmark.benchmark.monitoring import (ResourceMonitor,
                                                        StatisticsCalculator)
from hardware_ml_benchmark.benchmark.rendering import RenderingEngine
from hardware_ml_benchmark.core.cli import CommandLineInterface
# Import custom modules
from hardware_ml_benchmark.core.utils import (check_dependencies,
                                              print_dependency_status,
                                              setup_logging)
from hardware_ml_benchmark.data.datasets import DatasetLoader
from hardware_ml_benchmark.models.models import ModelLoader
from hardware_ml_benchmark.output.output import ResultExporter, Visualizer


class BenchmarkManager:
    """Benchmark test manager - main control class"""
    
    def __init__(self):
        # Setup logging system
        self.logger, self.log_filename = setup_logging()
        self.logger.info("=" * 80)
        self.logger.info("Deep Learning Benchmark Tool Starting")
        self.logger.info("=" * 80)
        
        # Check dependencies
        self.dependencies = check_dependencies()
        
        # Initialize CLI interface component
        self.cli_interface = CommandLineInterface(self.dependencies)
        
        # Initialize other components
        monitor_config = getattr(self, 'monitor_config', {})
        self.resource_monitor = ResourceMonitor(
            enable_gpu_monitoring=not monitor_config.get('disable_gpu_monitor', False),
            sample_interval=monitor_config.get('monitor_interval', 0.1),
            max_samples=monitor_config.get('monitor_samples', 1000)
        )
        self.stats_calculator = StatisticsCalculator()
        
        # Benchmark related objects
        self.dataset_loader = None
        self.model_loader = None
        self.rendering_engine = None
        self.benchmark_runner = None
        
        # Runtime data
        self.configuration = None
        self.model = None
        self.dataloader = None
        self.test_images = None
        
        self.logger.info("Benchmark tool initialization completed")
    
    def _log_test_configuration(self, config):
        """Log detailed test configuration information to log file"""
        self.logger.info("=" * 60)
        self.logger.info("Test Configuration Details")
        self.logger.info("=" * 60)
        
        # Basic configuration
        self.logger.info(f"Task: {config['task']}")
        self.logger.info(f"Model Name: {config['model_info']['name']}")
        self.logger.info(f"Model ID: {config['model_info']['model']}")
        if 'type' in config['model_info']:
            self.logger.info(f"Framework: {config['model_info']['type']}")
        
        # Dataset configuration
        self.logger.info(f"Dataset: {config['dataset_name']}")
        if config['test_samples'] == -1:
            self.logger.info(f"Test Samples: All samples")
        else:
            self.logger.info(f"Test Samples: {config['test_samples']}")
        
        # Computing device configuration
        self.logger.info(f"Device: {config['device']}")
        if config['device'].startswith('cuda'):
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.logger.info(f"GPU Device: {device_name}")
                self.logger.info(f"GPU Memory: {memory_gb:.1f}GB")
            else:
                self.logger.info("Warning: CUDA device specified but CUDA is not available")
        else:
            self.logger.info("Using CPU for computation")
        
        # Output configuration
        self.logger.info(f"Output Directory: {config['output_dir']}")
        self.logger.info(f"Generate Plots: {'yes' if config['plot'] else 'no'}")
        self.logger.info(f"Quiet Mode: {'yes' if config['quiet'] else 'no'}")
        
        self.logger.info("=" * 60)
    
    def _run_monitor_accuracy_test(self, args):
        """Run monitoring accuracy test"""
        print("MONITORING SYSTEM ACCURACY TEST")
        print("="*50)
        
        try:
            from benchmark.monitoring import MonitoringOverheadAnalyzer
            
            analyzer = MonitoringOverheadAnalyzer()
            
            # Use command line arguments
            duration = 10.0  # Fixed test duration
            sample_interval = getattr(args, 'monitor_interval', 0.1)
            
            print(f"Test Configuration:")
            print(f"  Test Duration: {duration} seconds")
            print(f"  Sampling Interval: {sample_interval} seconds")
            print(f"  GPU Monitoring: {'Disabled' if getattr(args, 'disable_gpu_monitor', False) else 'Enabled'}")
            print()
            
            overhead_stats = analyzer.measure_monitoring_overhead(
                duration=duration,
                sample_interval=sample_interval
            )
            
            if overhead_stats:
                analyzer.print_overhead_analysis(overhead_stats)
                
                # In quiet mode, only show key metrics
                if getattr(args, 'quiet', False):
                    print(f"RESULT: {overhead_stats['overhead']['relative_percent']:.2f}% overhead")
            else:
                print("Monitoring accuracy test failed")
                
        except ImportError as e:
            print(f"Import error: {e}")
            print("Please ensure monitoring.py file contains MonitoringOverheadAnalyzer class")
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Run main program"""
        try:
            # Parse command line arguments
            parser = self.cli_interface.create_parser()
            args = parser.parse_args()
            
            # Handle special commands
            if args.list_models:
                self.cli_interface.list_available_models()
                return
            
            if args.list_datasets:
                self.cli_interface.list_available_datasets()
                return
            
            if getattr(args, 'test_monitor_accuracy', False):
                self._run_monitor_accuracy_test(args)
                return
            
            # Validate basic parameters
            if not (args.task and args.model and args.dataset):
                print("Error: Must specify --task, --model and --dataset parameters")
                print("Use --help to see complete parameter description")
                print("Use --list-models to see available models")
                print("Use --list-datasets to see available datasets")
                print()
                print("Examples:")
                print("  python -m hardware_ml_benchmark.main --device cpu --task classification --model resnet18 --dataset MNIST ")
                print("  python -m hardware_ml_benchmark.main --device cuda:0 --task detection --model yolov8n --dataset Test-Images ")
                sys.exit(1)
            
            # Run command line mode
            self._run_cli_mode(args)
            
        except KeyboardInterrupt:
            self.logger.warning("Program interrupted by user")
            print("\nProgram interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during program execution: {e}")
            print(f"Error during program execution: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _run_cli_mode(self, args):
        """Run command line mode"""
        self.logger.info("Starting command line mode")
        
        # Validate parameters
        errors = self.cli_interface.validate_args(args)
        if errors:
            print("Parameter errors:")
            for error in errors:
                print(f"  - {error}")
            print("\nUse --help to see help information")
            print("Use --list-models to see available models")
            sys.exit(1)
        
        # Convert to configuration object
        self.configuration = self.cli_interface.args_to_config(args)
        
        # Log detailed test configuration to log file
        self._log_test_configuration(self.configuration)
        
        # Setup monitoring configuration
        self.monitor_config = {
            'disable_gpu_monitor': getattr(args, 'disable_gpu_monitor', False),
            'monitor_interval': getattr(args, 'monitor_interval', 0.1),
            'monitor_samples': getattr(args, 'monitor_samples', 1000)
        }
        
        # Ensure output directory exists - use absolute path
        output_dir = os.path.abspath(self.configuration['output_dir'])
        self.configuration['output_dir'] = output_dir
        
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                self.logger.info(f"Created output directory: {output_dir}")
            except Exception as e:
                self.logger.error(f"Unable to create output directory {output_dir}: {e}")
                print(f"Error: Unable to create output directory {output_dir}: {e}")
                sys.exit(1)
        
        # Print configuration summary
        self.cli_interface.print_config_summary(self.configuration)
        
        # Run benchmark
        self._run_benchmark_pipeline()
    
    def _run_benchmark_pipeline(self):
        """Run benchmark pipeline"""
        try:
            # Initialize components
            self._initialize_components()
            
            # Load dataset
            self._load_dataset()
            
            # Load model
            self._load_model()
            
            # Run benchmark
            self._run_benchmark()
            
        except Exception as e:
            self.logger.error(f"Benchmark pipeline execution failed: {e}")
            print(f"Benchmark pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _initialize_components(self):
        """Initialize all components"""
        self.logger.info("Initializing all components")
        
        try:
            # Initialize dataset loader
            self.dataset_loader = DatasetLoader(self.configuration['test_samples'])
            self.logger.info(f"Dataset loader initialized - target samples: {self.configuration['test_samples'] if self.configuration['test_samples'] != -1 else 'All'}")
            
            # Initialize model loader
            self.model_loader = ModelLoader(self.configuration['device'])
            self.logger.info(f"Model loader initialized - computing device: {self.configuration['device']}")
            
            # Initialize rendering engine
            self.rendering_engine = RenderingEngine(self.logger)
            self.logger.info("Rendering engine initialized")
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            print(f"Component initialization failed: {e}")
            raise e
    
    def _load_dataset(self):
        """Load dataset"""
        dataset_name = self.configuration['dataset_name']
        task = self.configuration['task']
        
        self.logger.info("=" * 50)
        self.logger.info(f"Starting to load dataset: {dataset_name} (for {task} task)")
        self.logger.info("=" * 50)
        
        if not self.configuration.get('quiet', False):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset: {dataset_name}")
        
        try:
            if task == 'classification':
                if dataset_name == 'MNIST':
                    self.dataloader = self.dataset_loader.load_mnist()
                elif dataset_name == 'CIFAR-10':
                    self.dataloader = self.dataset_loader.load_cifar10()
                elif dataset_name == 'ImageNet-Sample':
                    self.dataloader = self.dataset_loader.create_synthetic_classification_dataset()
                else:
                    raise ValueError(f"Unknown classification dataset: {dataset_name}")
            
            elif task == 'detection':
                if dataset_name == 'KITTI':
                    self.dataloader = self.dataset_loader.load_kitti()
                elif dataset_name in ['COCO-Sample', 'Test-Images']:
                    self.dataloader, self.test_images = self.dataset_loader.create_synthetic_detection_dataset()
                else:
                    raise ValueError(f"Unknown detection dataset: {dataset_name}")
            
            elif task == 'segmentation':
                if dataset_name == 'Cityscapes':
                    self.dataloader = self.dataset_loader.load_cityscapes()
                elif dataset_name == 'Synthetic-Segmentation':
                    self.dataloader = self.dataset_loader.create_synthetic_segmentation_dataset()
                else:
                    raise ValueError(f"Unknown segmentation dataset: {dataset_name}")
            
            self.logger.info(f"Dataset {dataset_name} loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
            print(f"Dataset loading failed: {e}")
            raise e
    
    def _load_model(self):
        """Load model"""
        model_info = self.configuration['model_info']
        task = self.configuration['task']
        device = self.configuration['device']
        
        self.logger.info("=" * 50)
        self.logger.info(f"Starting to load model: {model_info['name']} (task: {task}, device: {device})")
        self.logger.info("=" * 50)
        
        if not self.configuration.get('quiet', False):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading model: {model_info['name']}")
        
        try:
            self.model = self.model_loader.load_model(
                task,
                model_info
            )
            self.logger.info(f"Model {model_info['name']} loaded successfully, deployed to device {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_info['name']}: {e}")
            print(f"Model loading failed: {e}")
            raise e
    
    def _run_benchmark(self):
        """Run benchmark test"""
        model_info = self.configuration['model_info']
        task = self.configuration['task']
        test_samples = self.configuration['test_samples']
        
        self.logger.info("=" * 50)
        self.logger.info(f"Starting benchmark test")
        self.logger.info(f"Model: {model_info['name']} ({task})")
        self.logger.info(f"Dataset: {self.configuration['dataset_name']}")
        self.logger.info(f"Planned test samples: {test_samples if test_samples != -1 else 'All'}")
        self.logger.info("=" * 50)
        
        if not self.configuration.get('quiet', False):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running benchmark test")
        
        # Start resource monitoring
        monitor_thread = self.resource_monitor.start_monitoring()
        self.logger.info("System resource monitoring started")
        
        start_time = time.time()
        
        try:
            # Create benchmark runner
            self.benchmark_runner = BenchmarkRunner(
                model=self.model,
                model_type=task,
                model_info=model_info,
                device=self.configuration['device'],
                rendering_engine=self.rendering_engine,
                test_samples=test_samples
            )
            
            # Run corresponding benchmark type
            if task == 'classification':
                timing_results = self.benchmark_runner.run_classification_benchmark(self.dataloader)
            elif task == 'detection':
                timing_results = self.benchmark_runner.run_detection_benchmark(self.dataloader, self.test_images)
            elif task == 'segmentation':
                timing_results = self.benchmark_runner.run_segmentation_benchmark(self.dataloader)
            
            end_time = time.time()
            total_time = end_time - start_time
            actual_samples = self.benchmark_runner.total_samples
            
            self.logger.info("=" * 50)
            self.logger.info("Benchmark test completed")
            self.logger.info(f"Actual samples processed: {actual_samples}")
            self.logger.info(f"Total time: {total_time:.2f} seconds")
            self.logger.info(f"Average throughput: {actual_samples / total_time:.2f} samples/sec")
            self.logger.info("=" * 50)
            
            # Stop resource monitoring
            self.resource_monitor.stop_monitoring()
            self.logger.info("System resource monitoring stopped")
            
            # Get resource statistics
            resource_stats = self.resource_monitor.get_resource_stats()
            
            # Calculate statistics
            stats = self.stats_calculator.calculate_benchmark_statistics(
                timing_results=timing_results,
                total_time=total_time,
                total_samples=actual_samples,
                model_type=task,
                model_info=model_info,
                dataset_name=self.configuration['dataset_name'],
                device=self.configuration['device'],
                resource_stats=resource_stats
            )
            
            # Print results
            if not self.configuration.get('quiet', False):
                self.logger.info("=" * 50)
                self.logger.info("Starting to print test results")
                self.stats_calculator.print_results_summary(stats)
            
            # Save results and generate visualizations
            self._save_results_and_visualizations(stats)
            
        except KeyboardInterrupt:
            self.logger.warning("Test interrupted by user")
            print("\nTest interrupted by user")
            return None
        except Exception as e:
            self.logger.error(f"Error during testing: {e}")
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.resource_monitor.stop_monitoring()
    
    def _save_results_and_visualizations(self, stats):
        """Save results and generate visualizations"""
        task = self.configuration['task']
        output_dir = self.configuration.get('output_dir', './results')
        
        try:
            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.info(f"Created output directory: {output_dir}")
            
            # Create result exporter with correct output directory
            exporter = ResultExporter(
                detailed_results=self.benchmark_runner.detailed_results,
                results_dir=output_dir
            )
            
            # Save CSV results
            csv_filenames = exporter.save_detailed_csv_results(stats, task)
            
            # Create visualizations (if enabled)
            plot_files = []
            if self.configuration.get('plot', False):
                visualizer = Visualizer(
                    detailed_results=self.benchmark_runner.detailed_results,
                    results_dir=output_dir
                )
                plot_files = visualizer.create_visualizations(stats, task)
            else:
                self.logger.info("User has not enabled plot generation")
            
            # Print final result file information
            if not self.configuration.get('quiet', False):
                print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test completed!")
                print(f"Results saved in: {output_dir}")
                print(f"Log file: {self.log_filename}")
                if csv_filenames:
                    print(f"Detailed results file: {csv_filenames[0]}")
                    if len(csv_filenames) > 1:
                        print(f"Summary results file: {csv_filenames[1]}")
                
                if plot_files:
                    print("Generated plot files:")
                    for plot_file in plot_files:
                        print(f"  - {plot_file}")
                elif not self.configuration.get('plot', False):
                    print("Plot generation disabled (use --plot parameter to enable)")
            
            # In quiet mode, provide concise success message
            if self.configuration.get('quiet', False):
                print(f"SUCCESS: Results saved to {output_dir}")
                print(f"Throughput: {stats['performance']['throughput']:.2f} samples/sec")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            print(f"Error saving results: {e}")
            import traceback
            traceback.print_exc()
            raise e

def main():
    """Main function entry point"""
    benchmark_manager = BenchmarkManager()
    benchmark_manager.run()

if __name__ == "__main__":
    # add parent directory to sys.path to allow relative imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    main()