import os
import cv2
from typing import List, Dict
import json
import random
import logging

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, resize, to_pil_image
import numpy as np
import albumentations as A


class SAMSegDataset(Dataset):
    def __init__(self,
                 ann_files: List[str],
                 transform: Dict | None = None,
                 **kwargs):
        self.image_list = []
        self.mask_list = []
        for ann_file in ann_files:
            with open(ann_file, 'r') as f:
                meta_data = json.load(f)
            image_root = meta_data['info']['image_root']
            mask_root = meta_data['info']['mask_root']
            self.image_list += [os.path.join(image_root, item['file_name']) for item in meta_data['images']]
            self.mask_list += [os.path.join(mask_root, item['gt_file']) for item in meta_data['annotations']]
        
        self.imgsz = kwargs.get('imgsz', 512)
        if transform:
            crop_size = transform.get('crop_size', 512)
            min_s, max_s = int(crop_size *(1-transform.get('crop_ratio', 0))), int(crop_size *(1+transform.get('crop_ratio', 0)))
            self.transform = A.Compose([
                A.RandomSizedCrop(min_max_height=(min_s, max_s), height=self.imgsz, width=self.imgsz, p=1.0), # 随机裁剪
                A.HorizontalFlip(p=transform.get('flip_h', 0.5)),  # 水平翻转
                A.VerticalFlip(p=transform.get('flip_v', 0.5)), # 垂直翻转
                A.RandomBrightnessContrast(p=transform.get('random_bc', 0)),  # 随机调整亮度对比度
                A.HueSaturationValue(p=transform.get('random_h', 0)), # 随机调整色相、饱和度、亮度
            ])
        else:
            self.transform = None
    
        self.pixel_mean = torch.Tensor([106.7138, 70.5478, 50.7495]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([74.6907, 52.2190, 37.5536]).view(-1, 1, 1)
    
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return [newh, neww]
    
    def preprocess(self, image) -> torch.Tensor:
        """
        Expects a numpy with shape HxWxC
        """
        # scaling to 1024x1024
        img_size = 1024
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], img_size)

        image_scale = np.array(resize(to_pil_image(image), target_size))

        input_image_torch = torch.as_tensor(np.array(image_scale))
        # hxWxC -->  cxhxw
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
        self.image_scale_size = tuple(input_image_torch.shape[-2:])
        self.origin_size = tuple(image.shape[0:2])
        # Normalize colors
        input_image_torch = (input_image_torch - self.pixel_mean) / self.pixel_std

        # Pad to Cx1024x1024

        h, w = input_image_torch.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(input_image_torch, (0, padw, 0, padh))
        return x
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        gt_path = self.mask_list[idx]
        try:
            image_ori = cv2.imread(image_path)
            image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
            if image is None:
                raise ValueError(f"can not read file: {image_path}")
            elif gt_mask is None:
                raise ValueError(f"can not read file: {gt_path}")
        
            h, w = image.shape[:2]
            
            if self.transform is not None:
                augmented = self.transform(
                    image=image,
                    mask=gt_mask  
                )
                image = augmented['image']
                gt_mask= augmented['mask']  
             
            return {
                "image_original": image,
                "image_input": self.preprocess(image),
                "original_size": (h, w),
                "image_scale_size": self.image_scale_size,
                "image_path": image_path,
                "gt_mask": to_tensor(gt_mask),
            }
        
        except Exception as e:
            # 记录错误并返回None，后续在collate_fn中过滤
            print(f"跳过损坏文件: {str(e)}")
            return None 

