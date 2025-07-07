import torch
import numpy as np
import torch.nn.functional as F


def evaluate_segement(segmentation_result, ground_truth):
    # 将张量转换为布尔类型
    ground_truth = (ground_truth>0).bool()
    segmentation_result = (segmentation_result>0.5).bool()

    # 计算 TP, TN, FP, FN
    TP = (ground_truth & segmentation_result).sum().item()
    TN = (~ground_truth & ~segmentation_result).sum().item()
    FP = (~ground_truth & segmentation_result).sum().item()
    FN = (ground_truth & ~segmentation_result).sum().item()

    recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    dice = (2*TP) / (FP + 2*TP + FN) if (FP + 2*TP + FN) > 0  else 1.0
    iou = (TP) / (FP + TP + FN) if (FP + 2*TP + FN) > 0  else 1.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return {
        'DICE': dice,
        'IOU': iou,
        'RECALL': recall,
        'PRECISION': precision,
        'ACCURACY': accuracy,
    }

