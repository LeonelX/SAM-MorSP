from .sam import SAMMorSP
from .skeleton import SmoothSkeleton, SSFNet, MorSP
from core.registry import register_model 

def register_all_models():

    register_model('SAMMorSP')(SAMMorSP)
    register_model('SmoothSkeleton')(SmoothSkeleton)
    register_model('SSFNet')(SSFNet)
    register_model('MorSP')(MorSP)
    
register_all_models()

__all__ = ['SAMMorSP', 'SmoothSkeleton', 'SSFNet', 'MorSP']
