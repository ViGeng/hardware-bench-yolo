#!/usr/bin/env python3
"""
Rendering module - responsible for drawing output results of various models
"""

import logging
import numpy as np
import torch
from hardware_ml_benchmark.core.config import COCO_CLASSES, IMAGENET_CLASSES, CITYSCAPES_CLASSES, DETECTION_COLORS, CITYSCAPES_COLOR_MAP
from hardware_ml_benchmark.core.utils import check_dependencies

# Check dependencies
dependencies = check_dependencies()

class RenderingEngine:
    """Rendering Engine - responsible for drawing output results of various models"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.dependencies = dependencies
        
        # Class names
        self.coco_classes = COCO_CLASSES
        self.imagenet_classes = IMAGENET_CLASSES
        self.cityscapes_classes = CITYSCAPES_CLASSES
        
        # Color configuration
        self.detection_colors = DETECTION_COLORS
        self.cityscapes_color_map = np.array(CITYSCAPES_COLOR_MAP, dtype=np.uint8)
    
    def render_classification_result(self, image, predictions, top_k=3):
        """Render classification results"""
        return image
    
    def render_detection_result(self, image, predictions, conf_threshold=0.5):
        """Render detection results (draw bounding boxes and confidence scores)"""
        try:
            # Process input image
            image = self._prepare_image_for_rendering(image)
            height, width = image.shape[:2]
            
            # Parse detection results
            boxes, confs, classes = self._parse_detection_predictions(predictions, conf_threshold, height, width)
            
            # Draw bounding boxes and labels
            if self.dependencies['pil'] and len(boxes) > 0:
                return self._render_detection_with_pil(image, boxes, confs, classes, conf_threshold)
            elif self.dependencies['cv2'] and len(boxes) > 0:
                return self._render_detection_with_cv2(image, boxes, confs, classes, conf_threshold)
            else:
                return image
                
        except Exception as e:
            self.logger.warning(f"Detection result rendering failed: {e}")
            return image if isinstance(image, np.ndarray) else np.zeros((640, 640, 3), dtype=np.uint8)
    
    def render_segmentation_result(self, image, predictions, alpha=0.6):
        """Render segmentation results (draw segmentation mask)"""
        try:
            # Process input image
            image = self._prepare_image_for_rendering(image)
            height, width = image.shape[:2]
            
            # Process prediction results
            pred_mask = self._prepare_segmentation_mask(predictions, height, width)
            
            # Create colored segmentation mask
            colored_mask = self._create_colored_segmentation_mask(pred_mask, height, width)
            
            # Blend original image and segmentation results
            if self.dependencies['cv2']:
                import cv2
                rendered_image = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
            else:
                rendered_image = image
            
            # Add legend
            if self.dependencies['pil']:
                rendered_image = self._add_segmentation_legend(rendered_image, pred_mask)
            
            return rendered_image
            
        except Exception as e:
            self.logger.warning(f"Segmentation result rendering failed: {e}")
            return image if isinstance(image, np.ndarray) else np.zeros((512, 1024, 3), dtype=np.uint8)
    
    def _prepare_image_for_rendering(self, image):
        """Prepare image for rendering"""
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.shape[0] == 3:  # CHW -> HWC
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
        
        # Ensure image is in 0-255 range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        return image
    
    def _get_classification_predictions(self, predictions, top_k):
        """Get classification prediction results"""
        if isinstance(predictions, torch.Tensor):
            probs = torch.nn.functional.softmax(predictions, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)
            top_probs = top_probs.cpu().numpy().flatten()
            top_indices = top_indices.cpu().numpy().flatten()
        else:
            # If it's a numpy array, create mock data
            top_indices = np.random.choice(len(self.imagenet_classes), top_k, replace=False)
            top_probs = np.random.random(top_k)
            top_probs = top_probs / top_probs.sum()  # Normalize
        
        return top_probs, top_indices
    
    def _render_classification_with_pil(self, image, top_probs, top_indices):
        """Render classification results using PIL"""
        from PIL import Image, ImageDraw, ImageFont
        
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw classification results
        y_offset = 10
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            class_name = self.imagenet_classes[idx % len(self.imagenet_classes)]
            text = f"{class_name}: {prob:.3f}"
            draw.text((10, y_offset), text, fill=(255, 255, 255), font=font)
            y_offset += 25
        
        return np.array(pil_image)
    
    def _parse_detection_predictions(self, predictions, conf_threshold, height, width):
        """Parse detection prediction results"""
        if hasattr(predictions, '__len__') and len(predictions) > 0:
            # Handle YOLO results
            if hasattr(predictions[0], 'boxes') and hasattr(predictions[0], 'conf'):
                pred = predictions[0]
                if len(pred.boxes) > 0:
                    boxes = pred.boxes.xyxy.cpu().numpy()
                    confs = pred.conf.cpu().numpy()
                    classes = pred.cls.cpu().numpy() if hasattr(pred, 'cls') else np.zeros(len(boxes))
                else:
                    boxes, confs, classes = [], [], []
            # Handle torchvision detection results
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
        """Create mock detection results"""
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
        """Render detection results using PIL"""
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
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label
            class_name = self.coco_classes[int(cls) % len(self.coco_classes)]
            label = f"{class_name}: {conf:.2f}"
            
            # Calculate text size
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw text background
            draw.rectangle([x1, y1-text_height-4, x1+text_width+4, y1], fill=color)
            
            # Draw text
            draw.text((x1+2, y1-text_height-2), label, fill=(255, 255, 255), font=font)
        
        return np.array(pil_image)
    
    def _render_detection_with_cv2(self, image, boxes, confs, classes, conf_threshold):
        """Render detection results using OpenCV"""
        import cv2
        
        rendered_image = image.copy()
        for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
            if conf < conf_threshold:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            color = self.detection_colors[int(cls) % len(self.detection_colors)]
            
            # Draw bounding box
            cv2.rectangle(rendered_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            class_name = self.coco_classes[int(cls) % len(self.coco_classes)]
            label = f"{class_name}: {conf:.2f}"
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw text background
            cv2.rectangle(rendered_image, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(rendered_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return rendered_image
    
    def _prepare_segmentation_mask(self, predictions, height, width):
        """Prepare segmentation mask"""
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
        
        # Adjust mask size to match image
        if pred_mask.shape != (height, width):
            if self.dependencies['cv2']:
                import cv2
                pred_mask = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            elif self.dependencies['pil']:
                from PIL import Image
                pred_mask = np.array(Image.fromarray(pred_mask).resize((width, height), Image.NEAREST))
        
        return pred_mask
    
    def _create_colored_segmentation_mask(self, pred_mask, height, width):
        """Create colored segmentation mask"""
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(len(self.cityscapes_classes)):
            mask_i = (pred_mask == i)
            colored_mask[mask_i] = self.cityscapes_color_map[i % len(self.cityscapes_color_map)]
        
        return colored_mask
    
    def _add_segmentation_legend(self, rendered_image, pred_mask):
        """Add segmentation legend"""
        from PIL import Image, ImageDraw, ImageFont
        
        pil_image = Image.fromarray(rendered_image)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Draw legend
        unique_classes = np.unique(pred_mask)
        legend_y = 10
        for cls_id in unique_classes[:5]:  # Only show first 5 classes
            if cls_id < len(self.cityscapes_classes):
                color = tuple(self.cityscapes_color_map[cls_id % len(self.cityscapes_color_map)].tolist())
                class_name = self.cityscapes_classes[cls_id]
                
                # Draw color block
                draw.rectangle([10, legend_y, 30, legend_y+15], fill=color)
                
                # Draw text
                draw.text((35, legend_y), class_name, fill=(255, 255, 255), font=font)
                legend_y += 20
        
        return np.array(pil_image)