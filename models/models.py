#!/usr/bin/env python3
"""
Model module - responsible for loading and managing various deep learning models
"""

import logging
import time
import torch
import torchvision.models.detection as detection_models
from core.config import DETECTION_MODELS, CLASSIFICATION_MODELS, SEGMENTATION_MODELS
from core.utils import check_dependencies

# Check dependencies
dependencies = check_dependencies()

class ModelLoader:
    """Model loader class"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.dependencies = dependencies
    
    def load_classification_model(self, model_info):
        """Load classification model"""
        self.logger.info(f"Loading classification model using timm: {model_info['model']}")
        
        if not self.dependencies['timm']:
            raise ImportError("timm library not available. Install with: pip install timm")
        
        import timm
        
        model = timm.create_model(
            model_info['model'], 
            pretrained=True,
            num_classes=1000
        )
        model.eval()
        model = model.to(self.device)
        
        self.logger.info(f"Classification model loaded successfully: {model_info['name']}")
        return model
    
    def load_detection_model(self, model_info):
        """Load detection model"""
        if model_info['type'] == 'yolo':
            return self._load_yolo_model(model_info)
        elif model_info['type'] == 'torchvision':
            return self._load_torchvision_detection_model(model_info)
        else:
            raise ValueError(f"Unsupported detection model type: {model_info['type']}")
    
    def _load_yolo_model(self, model_info):
        """Load YOLO model"""
        if not self.dependencies['ultralytics']:
            raise ImportError("ultralytics library not available. Install with: pip install ultralytics")
        
        from ultralytics import YOLO
        
        self.logger.info(f"Loading detection model using YOLO: {model_info['model']}")
        
        model = YOLO(model_info['model'])
        
        self.logger.info(f"YOLO model loaded successfully: {model_info['name']}")
        return model
    
    def _load_torchvision_detection_model(self, model_info):
        """Load torchvision detection model"""
        if not self.dependencies['torchvision_detection']:
            raise ImportError("torchvision detection models not available")
        
        self.logger.info(f"Loading detection model using torchvision: {model_info['model']}")
        
        # Load torchvision detection model
        if model_info['model'] == 'fasterrcnn_resnet50_fpn':
            model = detection_models.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        elif model_info['model'] == 'fasterrcnn_mobilenet_v3_large_fpn':
            model = detection_models.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
        elif model_info['model'] == 'fcos_resnet50_fpn':
            model = detection_models.fcos_resnet50_fpn(weights='DEFAULT')
        else:
            raise ValueError(f"Unsupported torchvision detection model: {model_info['model']}")
        
        model.eval()
        model = model.to(self.device)
        
        self.logger.info(f"Torchvision detection model loaded successfully: {model_info['name']}")
        return model
    
    def load_segmentation_model(self, model_info):
        """Load segmentation model"""
        if not self.dependencies['smp']:
            raise ImportError("segmentation_models_pytorch not available. Install with: pip install segmentation-models-pytorch")
        
        import segmentation_models_pytorch as smp
        
        self.logger.info(f"Loading segmentation model using segmentation_models_pytorch: {model_info['model']}")
        
        # Create model using segmentation_models_pytorch
        model_class = getattr(smp, model_info['model'])
        model = model_class(
            encoder_name=model_info['encoder'],
            encoder_weights='imagenet',
            classes=19,  # Cityscapes has 19 classes
            activation=None
        )
        model.eval()
        model = model.to(self.device)
        
        self.logger.info(f"Segmentation model loaded successfully: {model_info['name']}")
        return model
    
    def load_model(self, model_type, model_info):
        """Load corresponding model based on model type"""
        self.logger.info(f"Starting to load model: {model_info['name']}")
        print(f"\nLoading model: {model_info['name']}...")
        
        try:
            if model_type == 'classification':
                model = self.load_classification_model(model_info)
            elif model_type == 'detection':
                model = self.load_detection_model(model_info)
            elif model_type == 'segmentation':
                model = self.load_segmentation_model(model_info)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.logger.info(f"Model loaded successfully: {model_info['name']}")
            print("Model loaded successfully!")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            print(f"Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            raise e

def get_available_models(model_type, dependencies):
    """Get list of available models"""
    if model_type == 'classification':
        if dependencies['timm']:
            return CLASSIFICATION_MODELS
        else:
            return {}
    
    elif model_type == 'detection':
        available_models = {}
        for key, value in DETECTION_MODELS.items():
            if value['type'] == 'yolo' and dependencies['ultralytics']:
                available_models[key] = value
            elif value['type'] == 'torchvision' and dependencies['torchvision_detection']:
                available_models[key] = value
        return available_models
    
    elif model_type == 'segmentation':
        if dependencies['smp']:
            return SEGMENTATION_MODELS
        else:
            return {}
    
    return {}

def validate_model_availability(model_type, dependencies):
    """Validate if model type is available"""
    if model_type == 'classification':
        return dependencies['timm']
    elif model_type == 'detection':
        return dependencies['ultralytics'] or dependencies['torchvision_detection']
    elif model_type == 'segmentation':
        return dependencies['smp']
    return False