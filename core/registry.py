"""
模块职责：集中管理所有可构建的类和函数
"""
from typing import Dict, Type
import torch.nn as nn

# 注册表字典
_loss_registry: Dict[str, Type[nn.Module]] = {}
_model_registry: Dict[str, Type[nn.Module]] = {}
_dataset_registry: Dict[str, Type[nn.Module]] = {}

def register_loss(name: str):
    """装饰器：注册损失函数"""
    def decorator(cls):
        _loss_registry[name] = cls
        return cls
    return decorator

def register_model(name: str):
    """装饰器：注册模型类"""
    def decorator(cls):
        _model_registry[name] = cls
        return cls
    return decorator

def register_dataset(name: str):
    """装饰器：注册数据集类"""
    def decorator(cls):
        _dataset_registry[name] = cls
        return cls
    return decorator

def get_loss_class(name: str) -> Type[nn.Module]:
    """获取已注册的损失类"""
    return _loss_registry[name]

def get_model_class(name: str) -> Type[nn.Module]:
    """获取已注册的模型类"""
    return _model_registry[name]

def get_dataset_class(name: str) -> Type[nn.Module]:
    """获取已注册的数据集类"""
    return _dataset_registry[name]
