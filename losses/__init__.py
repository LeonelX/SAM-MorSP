from .skeleton_loss import SoftCLDice, SkeletonEnergy


from core.registry import register_loss

# 显式注册所有损失函数
def register_all_losses():
    register_loss("SoftCLDice")(SoftCLDice)
    register_loss("SkeletonEnergy")(SkeletonEnergy)


register_all_losses()

__all__ = ['SoftCLDice', 'SkeletonEnergy']