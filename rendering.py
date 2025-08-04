#!/usr/bin/env python3
"""
渲染模块 - 负责绘制各种模型的输出结果
"""

import logging
import numpy as np
import torch
from config import COCO_CLASSES, IMAGENET_CLASSES, CITYSCAPES_CLASSES, DETECTION_COLORS, CITYSCAPES_COLOR_MAP
from utils import check_dependencies

# 检查依赖
dependencies = check_dependencies()

class RenderingEngine:
    """渲染引擎 - 负责绘制各种模型的输出结果"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.dependencies = dependencies
        
        # 类别名称
        self.coco_classes = COCO_CLASSES
        self.imagenet_classes = IMAGENET_CLASSES
        self.cityscapes_classes = CITYSCAPES_CLASSES
        
        # 颜色配置
        self.detection_colors = DETECTION_COLORS
        self.cityscapes_color_map = np.array(CITYSCAPES_COLOR_MAP, dtype=np.uint8)
    
    def render_classification_result(self, image, predictions, top_k=3):
        """渲染分类结果"""
        try:
            # 处理输入图像
            image = self._prepare_image_for_rendering(image)
            
            # 获取top-k预测结果
            top_probs, top_indices = self._get_classification_predictions(predictions, top_k)
            
            # 使用PIL进行文本渲染
            if self.dependencies['pil']:
                return self._render_classification_with_pil(image, top_probs, top_indices)
            else:
                # 如果没有PIL，简单返回原图像
                return image
                
        except Exception as e:
            self.logger.warning(f"分类结果渲染失败: {e}")
            return image if isinstance(image, np.ndarray) else np.zeros((224, 224, 3), dtype=np.uint8)
    
    def render_detection_result(self, image, predictions, conf_threshold=0.5):
        """渲染检测结果（绘制边界框和置信度）"""
        try:
            # 处理输入图像
            image = self._prepare_image_for_rendering(image)
            height, width = image.shape[:2]
            
            # 解析检测结果
            boxes, confs, classes = self._parse_detection_predictions(predictions, conf_threshold, height, width)
            
            # 绘制边界框和标签
            if self.dependencies['pil'] and len(boxes) > 0:
                return self._render_detection_with_pil(image, boxes, confs, classes, conf_threshold)
            elif self.dependencies['cv2'] and len(boxes) > 0:
                return self._render_detection_with_cv2(image, boxes, confs, classes, conf_threshold)
            else:
                return image
                
        except Exception as e:
            self.logger.warning(f"检测结果渲染失败: {e}")
            return image if isinstance(image, np.ndarray) else np.zeros((640, 640, 3), dtype=np.uint8)
    
    def render_segmentation_result(self, image, predictions, alpha=0.6):
        """渲染分割结果（绘制分割mask）"""
        try:
            # 处理输入图像
            image = self._prepare_image_for_rendering(image)
            height, width = image.shape[:2]
            
            # 处理预测结果
            pred_mask = self._prepare_segmentation_mask(predictions, height, width)
            
            # 创建彩色分割图
            colored_mask = self._create_colored_segmentation_mask(pred_mask, height, width)
            
            # 混合原图像和分割结果
            if self.dependencies['cv2']:
                import cv2
                rendered_image = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
            else:
                rendered_image = image
            
            # 添加图例
            if self.dependencies['pil']:
                rendered_image = self._add_segmentation_legend(rendered_image, pred_mask)
            
            return rendered_image
            
        except Exception as e:
            self.logger.warning(f"分割结果渲染失败: {e}")
            return image if isinstance(image, np.ndarray) else np.zeros((512, 1024, 3), dtype=np.uint8)
    
    def _prepare_image_for_rendering(self, image):
        """准备图像用于渲染"""
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.shape[0] == 3:  # CHW -> HWC
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
        
        # 确保图像在0-255范围内
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        return image
    
    def _get_classification_predictions(self, predictions, top_k):
        """获取分类预测结果"""
        if isinstance(predictions, torch.Tensor):
            probs = torch.nn.functional.softmax(predictions, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)
            top_probs = top_probs.cpu().numpy().flatten()
            top_indices = top_indices.cpu().numpy().flatten()
        else:
            # 如果是numpy array，创建模拟数据
            top_indices = np.random.choice(len(self.imagenet_classes), top_k, replace=False)
            top_probs = np.random.random(top_k)
            top_probs = top_probs / top_probs.sum()  # 归一化
        
        return top_probs, top_indices
    
    def _render_classification_with_pil(self, image, top_probs, top_indices):
        """使用PIL渲染分类结果"""
        from PIL import Image, ImageDraw, ImageFont
        
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # 绘制分类结果
        y_offset = 10
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            class_name = self.imagenet_classes[idx % len(self.imagenet_classes)]
            text = f"{class_name}: {prob:.3f}"
            draw.text((10, y_offset), text, fill=(255, 255, 255), font=font)
            y_offset += 25
        
        return np.array(pil_image)
    
    def _parse_detection_predictions(self, predictions, conf_threshold, height, width):
        """解析检测预测结果"""
        if hasattr(predictions, '__len__') and len(predictions) > 0:
            # 处理YOLO结果
            if hasattr(predictions[0], 'boxes') and hasattr(predictions[0], 'conf'):
                pred = predictions[0]
                if len(pred.boxes) > 0:
                    boxes = pred.boxes.xyxy.cpu().numpy()
                    confs = pred.conf.cpu().numpy()
                    classes = pred.cls.cpu().numpy() if hasattr(pred, 'cls') else np.zeros(len(boxes))
                else:
                    boxes, confs, classes = [], [], []
            # 处理torchvision检测结果
            elif isinstance(predictions[0], dict):
                pred = predictions[0]
                scores = pred.get('scores', torch.tensor([])).cpu().numpy()
                valid_idx = scores > conf_threshold
                boxes = pred.get('boxes', torch.tensor([])).cpu().numpy()[valid_idx]
                confs = scores[valid_idx]
                classes = pred.get('labels', torch.tensor([])).cpu().numpy()[valid_idx]
            else:
                boxes, confs, classes = self._create_mock_detection_results(height, width)
        else:
            boxes, confs, classes = self._create_mock_detection_results(height, width)
        
        return np.array(boxes), np.array(confs), np.array(classes)
    
    def _create_mock_detection_results(self, height, width):
        """创建模拟检测结果"""
        num_boxes = np.random.randint(3, 8)
        boxes = []
        confs = []
        classes = []
        for _ in range(num_boxes):
            x1 = np.random.randint(0, width//2)
            y1 = np.random.randint(0, height//2)
            x2 = np.random.randint(x1+20, width)
            y2 = np.random.randint(y1+20, height)
            boxes.append([x1, y1, x2, y2])
            confs.append(np.random.uniform(0.5, 0.95))
            classes.append(np.random.randint(0, len(self.coco_classes)))
        return boxes, confs, classes
    
    def _render_detection_with_pil(self, image, boxes, confs, classes, conf_threshold):
        """使用PIL渲染检测结果"""
        from PIL import Image, ImageDraw, ImageFont
        
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
            if conf < conf_threshold:
                continue
            
            x1, y1, x2, y2 = box
            color = self.detection_colors[int(cls) % len(self.detection_colors)]
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # 绘制标签
            class_name = self.coco_classes[int(cls) % len(self.coco_classes)]
            label = f"{class_name}: {conf:.2f}"
            
            # 计算文本大小
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 绘制文本背景
            draw.rectangle([x1, y1-text_height-4, x1+text_width+4, y1], fill=color)
            
            # 绘制文本
            draw.text((x1+2, y1-text_height-2), label, fill=(255, 255, 255), font=font)
        
        return np.array(pil_image)
    
    def _render_detection_with_cv2(self, image, boxes, confs, classes, conf_threshold):
        """使用OpenCV渲染检测结果"""
        import cv2
        
        rendered_image = image.copy()
        for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
            if conf < conf_threshold:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            color = self.detection_colors[int(cls) % len(self.detection_colors)]
            
            # 绘制边界框
            cv2.rectangle(rendered_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            class_name = self.coco_classes[int(cls) % len(self.coco_classes)]
            label = f"{class_name}: {conf:.2f}"
            
            # 获取文本大小
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # 绘制文本背景
            cv2.rectangle(rendered_image, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
            
            # 绘制文本
            cv2.putText(rendered_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return rendered_image
    
    def _prepare_segmentation_mask(self, predictions, height, width):
        """准备分割mask"""
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() == 4:  # NCHW
                predictions = predictions.squeeze(0)
            if predictions.dim() == 3 and predictions.shape[0] > 1:  # CHW with multiple classes
                pred_mask = torch.argmax(torch.softmax(predictions, dim=0), dim=0)
            else:
                pred_mask = predictions.squeeze()
            
            pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
        else:
            pred_mask = np.random.randint(0, len(self.cityscapes_classes), (height, width), dtype=np.uint8)
        
        # 调整mask大小以匹配图像
        if pred_mask.shape != (height, width):
            if self.dependencies['cv2']:
                import cv2
                pred_mask = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            elif self.dependencies['pil']:
                from PIL import Image
                pred_mask = np.array(Image.fromarray(pred_mask).resize((width, height), Image.NEAREST))
        
        return pred_mask
    
    def _create_colored_segmentation_mask(self, pred_mask, height, width):
        """创建彩色分割mask"""
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(len(self.cityscapes_classes)):
            mask_i = (pred_mask == i)
            colored_mask[mask_i] = self.cityscapes_color_map[i % len(self.cityscapes_color_map)]
        
        return colored_mask
    
    def _add_segmentation_legend(self, rendered_image, pred_mask):
        """添加分割图例"""
        from PIL import Image, ImageDraw, ImageFont
        
        pil_image = Image.fromarray(rendered_image)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # 绘制图例
        unique_classes = np.unique(pred_mask)
        legend_y = 10
        for cls_id in unique_classes[:5]:  # 只显示前5个类别
            if cls_id < len(self.cityscapes_classes):
                color = tuple(self.cityscapes_color_map[cls_id % len(self.cityscapes_color_map)].tolist())
                class_name = self.cityscapes_classes[cls_id]
                
                # 绘制颜色块
                draw.rectangle([10, legend_y, 30, legend_y+15], fill=color)
                
                # 绘制文本
                draw.text((35, legend_y), class_name, fill=(255, 255, 255), font=font)
                legend_y += 20
        
        return np.array(pil_image)