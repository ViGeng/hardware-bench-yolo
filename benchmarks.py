#!/usr/bin/env python3
"""
基准测试模块 - 包含各种模型类型的基准测试逻辑
"""

import time
import logging
import numpy as np
import torch
from utils import safe_time_value

class BenchmarkRunner:
    """基准测试运行器"""
    
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
    
    def run_classification_benchmark(self, dataloader):
        """运行分类模型基准测试"""
        self.logger.info("开始分类模型基准测试")
        print("\n开始分类模型基准测试...")
        print(f"正在使用模型: {self.model_info['name']}")
        print(f"计划测试样本数: {self.test_samples if self.test_samples != -1 else '全部'}")
        
        batch_times = []
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        rendering_times = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                batch_start = time.time()
                
                # 验证输入数据形状
                if batch_idx == 0:
                    self.logger.info(f"输入数据形状: {data.shape}, 数据类型: {data.dtype}")
                    print(f"输入数据形状: {data.shape}")
                    print(f"数据类型: {data.dtype}")
                    print(f"数据范围: [{data.min().item():.3f}, {data.max().item():.3f}]")
                
                # 预处理时间
                prep_start = time.time()
                data = data.to(self.device)
                prep_time = (time.time() - prep_start) * 1000  # ms
                
                # 推理时间
                inf_start = time.time()
                try:
                    output = self.model(data)
                    inf_time = (time.time() - inf_start) * 1000  # ms
                except Exception as e:
                    self.logger.error(f"推理过程中出错: {e}")
                    print(f"推理过程中出错: {e}")
                    raise e
                
                # 后处理时间（分类任务简单，几乎为0）
                post_start = time.time()
                post_time = (time.time() - post_start) * 1000  # ms
                
                # 渲染时间
                render_start = time.time()
                try:
                    rendered_image = self.rendering_engine.render_classification_result(data, output)
                    render_time = (time.time() - render_start) * 1000  # ms
                except Exception as e:
                    self.logger.warning(f"渲染失败: {e}")
                    render_time = 0.0
                
                batch_time = (time.time() - batch_start) * 1000  # ms
                
                # 安全处理时间值
                prep_time = safe_time_value(prep_time)
                inf_time = safe_time_value(inf_time)
                post_time = safe_time_value(post_time)
                render_time = safe_time_value(render_time)
                batch_time = safe_time_value(batch_time)
                
                # 记录详细结果
                self._record_batch_results(data, prep_time, inf_time, post_time, render_time, batch_time)
                
                # 记录汇总时间
                preprocessing_times.append(prep_time)
                inference_times.append(inf_time)
                postprocessing_times.append(post_time)
                rendering_times.append(render_time)
                batch_times.append(batch_time)
                
                self.total_samples += len(data)
                
                # 进度显示
                if batch_idx % 10 == 0:
                    fps = 1000.0 / (batch_time / len(data)) if batch_time > 0 else 0
                    self._print_progress(fps)
                
                # 检查是否达到目标样本数
                if self._should_stop_testing():
                    break
        
        self.logger.info(f"分类模型基准测试完成，总计处理 {self.total_samples} 个样本")
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'postprocessing_times': postprocessing_times,
            'rendering_times': rendering_times,
            'batch_times': batch_times
        }
    
    def run_detection_benchmark(self, dataloader, test_images=None):
        """运行检测模型基准测试"""
        self.logger.info("开始检测模型基准测试")
        print("\n开始检测模型基准测试...")
        print(f"计划测试图像数: {self.test_samples if self.test_samples != -1 else '全部'}")
        
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        rendering_times = []
        
        # 确定实际要测试的图像数量
        num_test_images = self._calculate_test_images_count(dataloader, test_images)
        
        self.logger.info(f"实际测试图像数: {num_test_images}")
        print(f"实际测试图像数: {num_test_images}")
        
        if self.model_info['type'] == 'yolo':
            return self._run_yolo_detection_benchmark(num_test_images)
        elif self.model_info['type'] == 'torchvision':
            return self._run_torchvision_detection_benchmark(dataloader, num_test_images)
    
    def run_segmentation_benchmark(self, dataloader):
        """运行分割模型基准测试"""
        self.logger.info("开始分割模型基准测试")
        print("\n开始分割模型基准测试...")
        print(f"正在使用模型: {self.model_info['name']}")
        print(f"计划测试样本数: {self.test_samples if self.test_samples != -1 else '全部'}")
        
        batch_times = []
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        rendering_times = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                batch_start = time.time()
                
                # 验证输入数据形状
                if batch_idx == 0:
                    self.logger.info(f"分割模型输入数据形状: {data.shape}, 目标形状: {target.shape}")
                    print(f"输入数据形状: {data.shape}")
                    print(f"目标形状: {target.shape}")
                
                # 预处理时间
                prep_start = time.time()
                data = data.to(self.device)
                prep_time = (time.time() - prep_start) * 1000  # ms
                
                # 推理时间
                inf_start = time.time()
                try:
                    output = self.model(data)
                    inf_time = (time.time() - inf_start) * 1000  # ms
                except Exception as e:
                    self.logger.error(f"分割推理过程中出错: {e}")
                    print(f"推理过程中出错: {e}")
                    raise e
                
                # 后处理时间（例如softmax和argmax）
                post_start = time.time()
                if output.dim() > 3:  # 如果输出是logits
                    pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
                else:
                    pred = output
                post_time = (time.time() - post_start) * 1000  # ms
                
                # 渲染时间
                render_start = time.time()
                try:
                    rendered_image = self.rendering_engine.render_segmentation_result(data, pred)
                    render_time = (time.time() - render_start) * 1000  # ms
                except Exception as e:
                    self.logger.warning(f"分割渲染失败: {e}")
                    render_time = 0.0
                
                batch_time = (time.time() - batch_start) * 1000  # ms
                
                # 安全处理时间值
                prep_time = safe_time_value(prep_time)
                inf_time = safe_time_value(inf_time)
                post_time = safe_time_value(post_time)
                render_time = safe_time_value(render_time)
                batch_time = safe_time_value(batch_time)
                
                # 记录详细结果
                self._record_batch_results(data, prep_time, inf_time, post_time, render_time, batch_time)
                
                # 记录汇总时间
                preprocessing_times.append(prep_time)
                inference_times.append(inf_time)
                postprocessing_times.append(post_time)
                rendering_times.append(render_time)
                batch_times.append(batch_time)
                
                self.total_samples += len(data)
                
                # 进度显示
                if batch_idx % 10 == 0:
                    fps = 1000.0 / (batch_time / len(data)) if batch_time > 0 else 0
                    self._print_progress(fps)
                
                # 检查是否达到目标样本数
                if self._should_stop_testing():
                    break
        
        self.logger.info(f"分割模型基准测试完成，总计处理 {self.total_samples} 个样本")
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'postprocessing_times': postprocessing_times,
            'rendering_times': rendering_times,
            'batch_times': batch_times
        }
    
    def _run_yolo_detection_benchmark(self, num_test_images):
        """运行YOLO检测基准测试"""
        self.logger.info("使用YOLO模型进行检测测试")
        
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        rendering_times = []
        
        for i in range(num_test_images):
            # 创建随机图像进行测试
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 记录总时间
            total_start = time.time()
            results = self.model(img, device=self.device, verbose=False)
            total_elapsed = (time.time() - total_start) * 1000  # ms
            
            # 获取时间信息
            prep_time, inf_time, post_time = self._extract_yolo_timing(results, total_elapsed)
            
            # 渲染时间
            render_start = time.time()
            try:
                rendered_image = self.rendering_engine.render_detection_result(img, results)
                render_time = (time.time() - render_start) * 1000  # ms
            except Exception as e:
                self.logger.warning(f"检测渲染失败: {e}")
                render_time = 0.0
            
            # 安全处理时间值
            prep_time = safe_time_value(prep_time)
            inf_time = safe_time_value(inf_time)
            post_time = safe_time_value(post_time)
            render_time = safe_time_value(render_time)
            
            total_time = prep_time + inf_time + post_time + render_time
            
            # 记录详细结果
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
            
            # 进度显示
            if i % 10 == 0 or i == num_test_images - 1:
                fps = 1000.0 / total_time if total_time > 0 else 0
                progress = ((i + 1) / num_test_images) * 100
                self.logger.info(f"YOLO检测进度: {i + 1}/{num_test_images} 图像 ({progress:.1f}%), 当前FPS: {fps:.1f}")
                print(f"Processed {i + 1}/{num_test_images} images ({progress:.1f}%)... 当前FPS: {fps:.1f}")
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'postprocessing_times': postprocessing_times,
            'rendering_times': rendering_times
        }
    
    def _run_torchvision_detection_benchmark(self, dataloader, num_test_images):
        """运行Torchvision检测基准测试"""
        self.logger.info("使用Torchvision模型进行检测测试")
        
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        rendering_times = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                # 预处理时间
                prep_start = time.time()
                data = data.to(self.device)
                if data.dim() == 4 and data.size(0) == 1:
                    data_list = [data.squeeze(0)]
                else:
                    data_list = [img for img in data]
                prep_time = (time.time() - prep_start) * 1000
                
                # 推理时间
                inf_start = time.time()
                try:
                    predictions = self.model(data_list)
                    inf_time = (time.time() - inf_start) * 1000
                except Exception as e:
                    self.logger.error(f"Torchvision检测推理过程中出错: {e}")
                    raise e
                
                # 后处理时间
                post_start = time.time()
                post_time = (time.time() - post_start) * 1000 + 1.0  # 假设后处理时间
                
                # 渲染时间
                render_start = time.time()
                try:
                    rendered_image = self.rendering_engine.render_detection_result(data_list[0], predictions)
                    render_time = (time.time() - render_start) * 1000
                except Exception as e:
                    self.logger.warning(f"Torchvision检测渲染失败: {e}")
                    render_time = 0.0
                
                total_time = prep_time + inf_time + post_time + render_time
                
                # 安全处理时间值
                prep_time = safe_time_value(prep_time)
                inf_time = safe_time_value(inf_time)
                post_time = safe_time_value(post_time)
                render_time = safe_time_value(render_time)
                total_time = safe_time_value(total_time)
                
                # 记录详细结果
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
                
                # 进度显示
                if batch_idx % 10 == 0:
                    fps = 1000.0 / total_time if total_time > 0 else 0
                    progress = (self.total_samples / num_test_images) * 100
                    self.logger.info(f"Torchvision检测进度: {self.total_samples}/{num_test_images} 图像 ({progress:.1f}%), 当前FPS: {fps:.1f}")
                    print(f"Processed {self.total_samples}/{num_test_images} images ({progress:.1f}%)... 当前FPS: {fps:.1f}")
                
                # 限制测试样本数
                if self.test_samples != -1 and self.total_samples >= self.test_samples:
                    break
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'postprocessing_times': postprocessing_times,
            'rendering_times': rendering_times
        }
    
    def _record_batch_results(self, data, prep_time, inf_time, post_time, render_time, batch_time):
        """记录批次详细结果"""
        batch_size = len(data)
        if batch_size > 0:
            for i in range(batch_size):
                sample_prep_time = prep_time / batch_size
                sample_inf_time = inf_time / batch_size
                sample_post_time = post_time / batch_size
                sample_render_time = render_time / batch_size
                sample_total_time = batch_time / batch_size
                
                # 确保每个样本时间都是合理的
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
        """提取YOLO时间信息"""
        prep_time = 0.0
        inf_time = total_elapsed  # 默认值
        post_time = 0.0
        
        if hasattr(results[0], 'speed'):
            speed = results[0].speed
            prep_time = speed.get('preprocess', 0)
            inf_time = speed.get('inference', 0)
            post_time = speed.get('postprocess', 0)
        
        return prep_time, inf_time, post_time
    
    def _calculate_test_images_count(self, dataloader, test_images):
        """计算实际测试图像数量"""
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
        """打印进度信息"""
        if self.test_samples == -1:
            self.logger.info(f"处理进度: {self.total_samples} 样本, 当前FPS: {fps:.1f}")
            print(f"Processed {self.total_samples} samples... 当前FPS: {fps:.1f}")
        else:
            progress = (self.total_samples / self.test_samples) * 100
            self.logger.info(f"处理进度: {self.total_samples}/{self.test_samples} 样本 ({progress:.1f}%), 当前FPS: {fps:.1f}")
            print(f"Processed {self.total_samples}/{self.test_samples} samples ({progress:.1f}%)... 当前FPS: {fps:.1f}")
    
    def _should_stop_testing(self):
        """检查是否应该停止测试"""
        if self.test_samples != -1 and self.total_samples >= self.test_samples:
            self.logger.info(f"达到目标样本数 {self.test_samples}，测试完成")
            print(f"达到目标样本数 {self.test_samples}，测试完成")
            return True
        return False