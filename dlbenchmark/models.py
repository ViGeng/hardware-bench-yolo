#!/usr/bin/env python3
"""
模型模块 - 负责加载和管理各种深度学习模型
"""

import logging
import time
import torch
import torchvision.models.detection as detection_models
from config import DETECTION_MODELS, CLASSIFICATION_MODELS, SEGMENTATION_MODELS
from utils import check_dependencies

# 检查依赖
dependencies = check_dependencies()

class ModelLoader:
    """模型加载器类"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.dependencies = dependencies
    
    def load_classification_model(self, model_info):
        """加载分类模型"""
        self.logger.info(f"使用timm加载分类模型: {model_info['model']}")
        
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
        
        self.logger.info(f"分类模型加载成功: {model_info['name']}")
        return model
    
    def load_detection_model(self, model_info):
        """加载检测模型"""
        if model_info['type'] == 'yolo':
            return self._load_yolo_model(model_info)
        elif model_info['type'] == 'torchvision':
            return self._load_torchvision_detection_model(model_info)
        else:
            raise ValueError(f"Unsupported detection model type: {model_info['type']}")
    
    def _load_yolo_model(self, model_info):
        """加载YOLO模型"""
        if not self.dependencies['ultralytics']:
            raise ImportError("ultralytics library not available. Install with: pip install ultralytics")
        
        from ultralytics import YOLO
        
        self.logger.info(f"使用YOLO加载检测模型: {model_info['model']}")
        
        model = YOLO(model_info['model'])
        
        self.logger.info(f"YOLO模型加载成功: {model_info['name']}")
        return model
    
    def _load_torchvision_detection_model(self, model_info):
        """加载torchvision检测模型"""
        if not self.dependencies['torchvision_detection']:
            raise ImportError("torchvision detection models not available")
        
        self.logger.info(f"使用torchvision加载检测模型: {model_info['model']}")
        
        # 加载torchvision检测模型
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
        
        self.logger.info(f"Torchvision检测模型加载成功: {model_info['name']}")
        return model
    
    def load_segmentation_model(self, model_info):
        """加载分割模型"""
        if not self.dependencies['smp']:
            raise ImportError("segmentation_models_pytorch not available. Install with: pip install segmentation-models-pytorch")
        
        import segmentation_models_pytorch as smp
        
        self.logger.info(f"使用segmentation_models_pytorch加载分割模型: {model_info['model']}")
        
        # 使用segmentation_models_pytorch创建模型
        model_class = getattr(smp, model_info['model'])
        model = model_class(
            encoder_name=model_info['encoder'],
            encoder_weights='imagenet',
            classes=19,  # Cityscapes有19个类别
            activation=None
        )
        model.eval()
        model = model.to(self.device)
        
        self.logger.info(f"分割模型加载成功: {model_info['name']}")
        return model
    
    def load_model(self, model_type, model_info):
        """根据模型类型加载相应模型"""
        self.logger.info(f"开始加载模型: {model_info['name']}")
        print(f"\n正在加载模型: {model_info['name']}...")
        
        try:
            if model_type == 'classification':
                model = self.load_classification_model(model_info)
            elif model_type == 'detection':
                model = self.load_detection_model(model_info)
            elif model_type == 'segmentation':
                model = self.load_segmentation_model(model_info)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.logger.info(f"模型加载成功: {model_info['name']}")
            print("模型加载成功!")
            
            return model
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise e

def get_available_models(model_type, dependencies):
    """获取可用的模型列表"""
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
    """验证模型类型是否可用"""
    if model_type == 'classification':
        return dependencies['timm']
    elif model_type == 'detection':
        return dependencies['ultralytics'] or dependencies['torchvision_detection']
    elif model_type == 'segmentation':
        return dependencies['smp']
    return False