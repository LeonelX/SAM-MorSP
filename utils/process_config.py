import os
import importlib.util
import sys
from pathlib import Path
from copy import deepcopy

def load_config_as_dict(config_path):
    """加载配置文件并处理_base_继承
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 合并后的配置字典
    """
    config_path = Path(config_path).absolute()
    config_dir = config_path.parent
    
    # 1. 首先加载基础配置
    base_config = {}
    current_config = _load_single_config(config_path)
    
    if 'base' in current_config:
        for base_file in current_config['base']:
            base_file_path = (config_dir / base_file).resolve()
            base_config = _merge_dict(base_config, load_config_as_dict(base_file_path))
    
    # 2. 合并当前配置
    final_config = _merge_dict(base_config, current_config)
    return final_config


def export_config(config_dict, output_path):
    """将配置字典导出为Python配置文件
    
    Args:
        config_dict: 包含配置参数的字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Auto-generated configuration file\n\n")  # 文件头注释
        
        # 导出简单变量（基础类型）
        for key, value in config_dict.items():
            if isinstance(value, (int, float, str, bool)):
                f.write(f"{key} = {repr(value)}\n\n")
        
        # 导出复杂结构（嵌套字典）
        for section in ['model', 'data', 'optimizer']: 
            if section in config_dict:
                f.write(f"{section} = {format_value(config_dict[section])}\n\n")
                

def _load_single_config(filepath):
    """加载单个配置文件为字典"""
    filepath = Path(filepath).absolute()
    module_name = f"config_{filepath.stem}"
    
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return {
        k: v for k, v in module.__dict__.items()
        if not k.startswith('_') and not k.endswith('_') 
    }


def _merge_dict(base, update):
    """深度合并两个字典"""
    result = deepcopy(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def format_value(value, indent=0):
    """递归格式化配置值"""
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            items.append(f"{' '*(indent+4)}{repr(k)}: {format_value(v, indent+4)}")
        return "{\n" + ",\n".join(items) + "\n" + " "*indent + "}"
    elif isinstance(value, list):
        if all(isinstance(x, dict) for x in value):
            return "[\n" + ",\n".join(format_value(x, indent+4) for x in value) + "\n" + " "*indent + "]"
        else:
            return repr(value)
    else:
        return repr(value)



                
