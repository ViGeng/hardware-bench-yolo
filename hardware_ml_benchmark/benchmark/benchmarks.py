#!/usr/bin/env python3
"""
Benchmark test module - contains benchmark test logic for various model types
"""

import time
import logging
import numpy as np
import torch
from hardware_ml_benchmark.core.utils import safe_time_value, check_dependencies

# Check dependencies
dependencies = check_dependencies()

class BenchmarkRunner:
    """Benchmark test runner"""
    
    def __init__(self, model, model_type, model_info, device, rendering_engine, test_samples=100):
        self.model = model
        self.model_type = model_type
        self.model_info = model_info
        self.device = device
        self.rendering_engine = rendering_engine
        self.test_samples = test_samples
        self.logger = logging.getLogger(__name__)
        
        self.total_samples = 0
        self.detailed_results = []
        
        # Initialize tqdm (if available)
        self.tqdm_available = dependencies.get('tqdm', False)
        if self.tqdm_available:
            try:
                from tqdm import tqdm
                self.tqdm = tqdm
            except ImportError:
                self.tqdm_available = False
                self.tqdm = None
        else:
            self.tqdm = None
    
    def run_classification_benchmark(self, dataloader):
        """Run classification model benchmark test"""
        self.logger.info("Starting classification model benchmark test")
        print("\nStarting classification model benchmark test...")
        print(f"Using model: {self.model_info['name']}")
        print(f"Planned test samples: {self.test_samples if self.test_samples != -1 else 'All'}")
        
        batch_times = []
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        rendering_times = []
        
        self.model.eval()
        
        # Calculate total iterations - modified to support unlimited sampling
        if self.test_samples == -1:
            total_iterations = len(dataloader)
        else:
            total_iterations = self.test_samples
        
        # Use tqdm or traditional progress display
        if self.tqdm_available:
            progress_bar = self.tqdm(
                total=total_iterations,
                desc="Processing samples",
                unit="samples",
                disable=False
            )
        else:
            progress_bar = None
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                batch_start = time.time()
                
                # Verify input data shape
                if batch_idx == 0:
                    self.logger.info(f"Input data shape: {data.shape}, data type: {data.dtype}")
                    print(f"Input data shape: {data.shape}")
                    print(f"Data type: {data.dtype}")
                    print(f"Data range: [{data.min().item():.3f}, {data.max().item():.3f}]")
                
                # Preprocessing time
                prep_start = time.time()
                data = data.to(self.device)
                prep_time = (time.time() - prep_start) * 1000  # ms
                
                # Inference time
                inf_start = time.time()
                try:
                    output = self.model(data)
                    inf_time = (time.time() - inf_start) * 1000  # ms
                except Exception as e:
                    self.logger.error(f"Error during inference: {e}")
                    print(f"Error during inference: {e}")
                    raise e
                
                # Postprocessing time (classification task is simple, almost 0)
                post_start = time.time()
                post_time = (time.time() - post_start) * 1000  # ms
                
                # Rendering time
                render_start = time.time()
                try:
                    rendered_image = self.rendering_engine.render_classification_result(data, output)
                    render_time = (time.time() - render_start) * 1000  # ms
                except Exception as e:
                    self.logger.warning(f"Rendering failed: {e}")
                    render_time = 0.0
                
                batch_time = (time.time() - batch_start) * 1000  # ms
                
                # Safely handle time values
                prep_time = safe_time_value(prep_time)
                inf_time = safe_time_value(inf_time)
                post_time = safe_time_value(post_time)
                render_time = safe_time_value(render_time)
                batch_time = safe_time_value(batch_time)
                
                # Record detailed results
                self._record_batch_results(data, prep_time, inf_time, post_time, render_time, batch_time)
                
                # Record aggregated times
                preprocessing_times.append(prep_time)
                inference_times.append(inf_time)
                postprocessing_times.append(post_time)
                rendering_times.append(render_time)
                batch_times.append(batch_time)
                
                self.total_samples += len(data)
                
                # Update progress bar information
                if self.tqdm_available and progress_bar:
                    fps = 1000.0 / (batch_time / len(data)) if batch_time > 0 else 0
                    progress_bar.set_postfix({'FPS': f'{fps:.1f}'})
                    progress_bar.update(len(data))
                else:
                    # Traditional progress display (reduced frequency)
                    if batch_idx % 50 == 0:
                        fps = 1000.0 / (batch_time / len(data)) if batch_time > 0 else 0
                        self._print_progress(fps)
                
                # Check if target sample count is reached
                if self._should_stop_testing():
                    break
        
        # Close progress bar
        if self.tqdm_available and progress_bar:
            progress_bar.close()
        
        self.logger.info(f"Classification model benchmark test completed, processed {self.total_samples} samples in total")
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'postprocessing_times': postprocessing_times,
            'rendering_times': rendering_times,
            'batch_times': batch_times
        }
    
    def run_detection_benchmark(self, dataloader, test_images=None):
        """Run detection model benchmark test"""
        self.logger.info("Starting detection model benchmark test")
        print("\nStarting detection model benchmark test...")
        print(f"Planned test images: {self.test_samples if self.test_samples != -1 else 'All'}")
        
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        rendering_times = []
        
        # Determine actual number of images to test
        num_test_images = self._calculate_test_images_count(dataloader, test_images)
        
        self.logger.info(f"Actual test image count: {num_test_images}")
        print(f"Actual test image count: {num_test_images}")
        
        if self.model_info['type'] == 'yolo':
            return self._run_yolo_detection_benchmark(num_test_images)
        elif self.model_info['type'] == 'torchvision':
            return self._run_torchvision_detection_benchmark(dataloader, num_test_images)
    
    def run_segmentation_benchmark(self, dataloader):
        """Run segmentation model benchmark test"""
        self.logger.info("Starting segmentation model benchmark test")
        print("\nStarting segmentation model benchmark test...")
        print(f"Using model: {self.model_info['name']}")
        print(f"Planned test samples: {self.test_samples if self.test_samples != -1 else 'All'}")
        
        batch_times = []
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        rendering_times = []
        
        self.model.eval()
        
        # Calculate total iterations - modified to support unlimited sampling
        if self.test_samples == -1:
            total_iterations = len(dataloader)
        else:
            total_iterations = self.test_samples
        
        # Use tqdm or traditional progress display
        if self.tqdm_available:
            progress_bar = self.tqdm(
                total=total_iterations,
                desc="Processing samples",
                unit="samples",
                disable=False
            )
        else:
            progress_bar = None
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                batch_start = time.time()
                
                # Verify input data shape
                if batch_idx == 0:
                    self.logger.info(f"Segmentation model input data shape: {data.shape}, target shape: {target.shape}")
                    print(f"Input data shape: {data.shape}")
                    print(f"Target shape: {target.shape}")
                
                # Preprocessing time
                prep_start = time.time()
                data = data.to(self.device)
                prep_time = (time.time() - prep_start) * 1000  # ms
                
                # Inference time
                inf_start = time.time()
                try:
                    output = self.model(data)
                    inf_time = (time.time() - inf_start) * 1000  # ms
                except Exception as e:
                    self.logger.error(f"Error during segmentation inference: {e}")
                    print(f"Error during inference: {e}")
                    raise e
                
                # Postprocessing time (e.g., softmax and argmax)
                post_start = time.time()
                if output.dim() > 3:  # If output is logits
                    pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
                else:
                    pred = output
                post_time = (time.time() - post_start) * 1000  # ms
                
                # Rendering time
                render_start = time.time()
                try:
                    rendered_image = self.rendering_engine.render_segmentation_result(data, pred)
                    render_time = (time.time() - render_start) * 1000  # ms
                except Exception as e:
                    self.logger.warning(f"Segmentation rendering failed: {e}")
                    render_time = 0.0
                
                batch_time = (time.time() - batch_start) * 1000  # ms
                
                # Safely handle time values
                prep_time = safe_time_value(prep_time)
                inf_time = safe_time_value(inf_time)
                post_time = safe_time_value(post_time)
                render_time = safe_time_value(render_time)
                batch_time = safe_time_value(batch_time)
                
                # Record detailed results
                self._record_batch_results(data, prep_time, inf_time, post_time, render_time, batch_time)
                
                # Record aggregated times
                preprocessing_times.append(prep_time)
                inference_times.append(inf_time)
                postprocessing_times.append(post_time)
                rendering_times.append(render_time)
                batch_times.append(batch_time)
                
                self.total_samples += len(data)
                
                # Update progress bar information
                if self.tqdm_available and progress_bar:
                    fps = 1000.0 / (batch_time / len(data)) if batch_time > 0 else 0
                    progress_bar.set_postfix({'FPS': f'{fps:.1f}'})
                    progress_bar.update(len(data))
                else:
                    # Traditional progress display (reduced frequency)
                    if batch_idx % 50 == 0:
                        fps = 1000.0 / (batch_time / len(data)) if batch_time > 0 else 0
                        self._print_progress(fps)
                
                # Check if target sample count is reached
                if self._should_stop_testing():
                    break
        
        # Close progress bar
        if self.tqdm_available and progress_bar:
            progress_bar.close()
        
        self.logger.info(f"Segmentation model benchmark test completed, processed {self.total_samples} samples in total")
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'postprocessing_times': postprocessing_times,
            'rendering_times': rendering_times,
            'batch_times': batch_times
        }
    
    def _run_yolo_detection_benchmark(self, num_test_images):
        """Run YOLO detection benchmark test"""
        self.logger.info("Using YOLO model for detection testing")
        
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        rendering_times = []
        
        # Use tqdm or traditional progress display
        if self.tqdm_available:
            progress_bar = self.tqdm(
                range(num_test_images),
                desc="Processing images",
                unit="images",
                disable=False
            )
        else:
            progress_bar = range(num_test_images)
        
        for i in progress_bar:
            # Create random image for testing
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Record total time
            total_start = time.time()
            results = self.model(img, device=self.device, verbose=False)
            total_elapsed = (time.time() - total_start) * 1000  # ms
            
            # Get timing information
            prep_time, inf_time, post_time = self._extract_yolo_timing(results, total_elapsed)
            
            # Rendering time
            render_start = time.time()
            try:
                rendered_image = self.rendering_engine.render_detection_result(img, results)
                render_time = (time.time() - render_start) * 1000  # ms
            except Exception as e:
                self.logger.warning(f"Detection rendering failed: {e}")
                render_time = 0.0
            
            # Safely handle time values
            prep_time = safe_time_value(prep_time)
            inf_time = safe_time_value(inf_time)
            post_time = safe_time_value(post_time)
            render_time = safe_time_value(render_time)
            
            total_time = prep_time + inf_time + post_time + render_time
            
            # Record detailed results
            self.detailed_results.append({
                'sample_id': i,
                'preprocessing_time': prep_time,
                'inference_time': inf_time,
                'postprocessing_time': post_time,
                'rendering_time': render_time,
                'total_time': total_time
            })
            
            preprocessing_times.append(prep_time)
            inference_times.append(inf_time)
            postprocessing_times.append(post_time)
            rendering_times.append(render_time)
            
            self.total_samples += 1
            
            # Update progress bar information
            if self.tqdm_available:
                fps = 1000.0 / total_time if total_time > 0 else 0
                progress_bar.set_postfix({'FPS': f'{fps:.1f}'})
            else:
                # Traditional progress display (reduced frequency)
                if i % 50 == 0 or i == num_test_images - 1:
                    fps = 1000.0 / total_time if total_time > 0 else 0
                    progress = ((i + 1) / num_test_images) * 100
                    self.logger.info(f"YOLO detection progress: {i + 1}/{num_test_images} images ({progress:.1f}%), current FPS: {fps:.1f}")
                    print(f"Processed {i + 1}/{num_test_images} images ({progress:.1f}%)... current FPS: {fps:.1f}")
        
        # Close progress bar
        if self.tqdm_available:
            progress_bar.close()
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'postprocessing_times': postprocessing_times,
            'rendering_times': rendering_times
        }
    
    def _run_torchvision_detection_benchmark(self, dataloader, num_test_images):
        """Run Torchvision detection benchmark test"""
        self.logger.info("Using Torchvision model for detection testing")
        
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        rendering_times = []
        
        self.model.eval()
        
        # Calculate total iterations
        if self.test_samples == -1:
            total_iterations = len(dataloader)
        else:
            total_iterations = min(num_test_images, self.test_samples)
        
        # Use tqdm or traditional progress display
        if self.tqdm_available:
            progress_bar = self.tqdm(
                total=total_iterations,
                desc="Processing images",
                unit="images",
                disable=False
            )
        else:
            progress_bar = None
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                # Preprocessing time
                prep_start = time.time()
                data = data.to(self.device)
                if data.dim() == 4 and data.size(0) == 1:
                    data_list = [data.squeeze(0)]
                else:
                    data_list = [img for img in data]
                prep_time = (time.time() - prep_start) * 1000
                
                # Inference time
                inf_start = time.time()
                try:
                    predictions = self.model(data_list)
                    inf_time = (time.time() - inf_start) * 1000
                except Exception as e:
                    self.logger.error(f"Error during Torchvision detection inference: {e}")
                    raise e
                
                # Postprocessing time
                post_start = time.time()
                post_time = (time.time() - post_start) * 1000 + 1.0  # Assume postprocessing time
                
                # Rendering time
                render_start = time.time()
                try:
                    rendered_image = self.rendering_engine.render_detection_result(data_list[0], predictions)
                    render_time = (time.time() - render_start) * 1000
                except Exception as e:
                    self.logger.warning(f"Torchvision detection rendering failed: {e}")
                    render_time = 0.0
                
                total_time = prep_time + inf_time + post_time + render_time
                
                # Safely handle time values
                prep_time = safe_time_value(prep_time)
                inf_time = safe_time_value(inf_time)
                post_time = safe_time_value(post_time)
                render_time = safe_time_value(render_time)
                total_time = safe_time_value(total_time)
                
                # Record detailed results
                self.detailed_results.append({
                    'sample_id': batch_idx,
                    'preprocessing_time': prep_time,
                    'inference_time': inf_time,
                    'postprocessing_time': post_time,
                    'rendering_time': render_time,
                    'total_time': total_time
                })
                
                preprocessing_times.append(prep_time)
                inference_times.append(inf_time)
                postprocessing_times.append(post_time)
                rendering_times.append(render_time)
                
                self.total_samples += 1
                
                # Update progress bar information
                if self.tqdm_available and progress_bar:
                    fps = 1000.0 / total_time if total_time > 0 else 0
                    progress_bar.set_postfix({'FPS': f'{fps:.1f}'})
                    progress_bar.update(1)
                else:
                    # Traditional progress display (reduced frequency)
                    if batch_idx % 50 == 0:
                        fps = 1000.0 / total_time if total_time > 0 else 0
                        progress = (self.total_samples / num_test_images) * 100
                        self.logger.info(f"Torchvision detection progress: {self.total_samples}/{num_test_images} images ({progress:.1f}%), current FPS: {fps:.1f}")
                        print(f"Processed {self.total_samples}/{num_test_images} images ({progress:.1f}%)... current FPS: {fps:.1f}")
                
                # Limit test sample count
                if self.test_samples != -1 and self.total_samples >= self.test_samples:
                    break
        
        # Close progress bar
        if self.tqdm_available and progress_bar:
            progress_bar.close()
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'postprocessing_times': postprocessing_times,
            'rendering_times': rendering_times
        }
    
    def _record_batch_results(self, data, prep_time, inf_time, post_time, render_time, batch_time):
        """Record batch detailed results"""
        batch_size = len(data)
        if batch_size > 0:
            for i in range(batch_size):
                sample_prep_time = prep_time / batch_size
                sample_inf_time = inf_time / batch_size
                sample_post_time = post_time / batch_size
                sample_render_time = render_time / batch_size
                sample_total_time = batch_time / batch_size
                
                # Ensure each sample time is reasonable
                sample_prep_time = safe_time_value(sample_prep_time)
                sample_inf_time = safe_time_value(sample_inf_time)
                sample_post_time = safe_time_value(sample_post_time)
                sample_render_time = safe_time_value(sample_render_time)
                sample_total_time = safe_time_value(sample_total_time)
                
                self.detailed_results.append({
                    'sample_id': self.total_samples + i,
                    'preprocessing_time': sample_prep_time,
                    'inference_time': sample_inf_time,
                    'postprocessing_time': sample_post_time,
                    'rendering_time': sample_render_time,
                    'total_time': sample_total_time
                })
    
    def _extract_yolo_timing(self, results, total_elapsed):
        """Extract YOLO timing information"""
        prep_time = 0.0
        inf_time = total_elapsed  # Default value
        post_time = 0.0
        
        if hasattr(results[0], 'speed'):
            speed = results[0].speed
            prep_time = speed.get('preprocess', 0)
            inf_time = speed.get('inference', 0)
            post_time = speed.get('postprocess', 0)
        
        return prep_time, inf_time, post_time
    
    def _calculate_test_images_count(self, dataloader, test_images):
        """Calculate actual test image count"""
        if hasattr(self, 'test_images') or test_images:
            available_images = len(test_images) if test_images else len(self.test_images)
            if self.test_samples == -1:
                return available_images
            else:
                return min(self.test_samples, available_images)
        else:
            dataset_size = len(dataloader.dataset)
            if self.test_samples == -1:
                return dataset_size
            else:
                return min(self.test_samples, dataset_size)
    
    def _print_progress(self, fps):
        """Print progress information (only used when tqdm is not available)"""
        if self.test_samples == -1:
            self.logger.info(f"Processing progress: {self.total_samples} samples, current FPS: {fps:.1f}")
            print(f"Processed {self.total_samples} samples... current FPS: {fps:.1f}")
        else:
            progress = (self.total_samples / self.test_samples) * 100
            self.logger.info(f"Processing progress: {self.total_samples}/{self.test_samples} samples ({progress:.1f}%), current FPS: {fps:.1f}")
            print(f"Processed {self.total_samples}/{self.test_samples} samples ({progress:.1f}%)... current FPS: {fps:.1f}")
    
    def _should_stop_testing(self):
        """Check if testing should be stopped"""
        if self.test_samples != -1 and self.total_samples >= self.test_samples:
            self.logger.info(f"Reached target sample count {self.test_samples}, testing completed")
            if not self.tqdm_available:
                print(f"Reached target sample count {self.test_samples}, testing completed")
            return True
        return False