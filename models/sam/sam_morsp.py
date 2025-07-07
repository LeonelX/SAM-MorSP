import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
import torchvision.transforms as T
import torch.nn.functional as F
from segment_anything import sam_model_registry

from core.registry import register_model
from losses import *
from builder import build_loss, build_model
from .sam_lora import LoRA_sam

@register_model("SAMMorSP")
class SAMMorSP(nn.Module):
    def __init__(
            self,
            sam_type: str = "vit_b",
            sam_weights: str = None,
            lora_rank: int = 4,
            MorSP: Dict = None,
            loss_mask: Dict = dict(type="BCEWithLogitsLoss"),
            loss_skel: Dict = dict(type="SoftCLDice", loss_weight=0.1),
            **kwargs
    ):
        super(SAMMorSP, self).__init__()
        sam_model = sam_model_registry[sam_type](checkpoint=sam_weights)
        self.lora_sam = LoRA_sam(sam_model, lora_rank)
        
        self.image_encoder = self.lora_sam.sam.image_encoder
        self.mask_decoder = self.lora_sam.sam.mask_decoder
        self.prompt_encoder = self.lora_sam.sam.prompt_encoder
        
        self.loss_mask = build_loss(loss_mask)
        self.loss_skel = build_loss(loss_skel)
                
        if MorSP is None:
            self.MorSP_module = None
        else:
            self.MorSP_module = build_model(MorSP)
            
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        
        # freeze image encoder
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.size befor padding after scaling
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0][0], : input_size[1][0]]
        masks = F.interpolate(masks, (original_size[0][0], original_size[1][0]),
                              mode="bilinear", align_corners=False)
        return masks
    
    def get_input(self, batch_input):
        input_image = batch_input["image_input"]
        image_original = batch_input["image_original"]
        image_scale_size = tuple(batch_input["image_scale_size"])
        original_size = tuple(batch_input["original_size"])
        
        with torch.no_grad():
            if image_original.shape[-1] == 4: # RGBN
                mask_prompt = F.interpolate(T.ToTensor()(image_original[..., -1]).to(input_image.device).unsqueeze(1),
                                            size=(256, 256),
                                            mode='bilinear',
                                            align_corners=False)
            else:
                mask_prompt = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=mask_prompt,
            ) # 提示编码
        
        return input_image, sparse_embeddings, dense_embeddings, image_scale_size, original_size
    
    def forward_train(self, batch_input):
        input_image, sparse_embeddings, dense_embeddings, scale_size, original_size = self.get_input(batch_input)
        # 图像编码
        image_embedding = self.image_encoder(input_image)  # (N, 256, 64, 64)
        # 解码获得低分辨率掩码
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (N, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (0, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (0, 256, 64, 64)
            multimask_output=True,
        )
        # 上采样
        outputs = self.postprocess_masks(low_res_masks,
                                        scale_size,
                                        original_size) # (N, 3, 1024, 1024)
        del image_embedding, low_res_masks # 删除无用变量
        # 骨架STD处理特征
        o = outputs[:, 0, :, :].unsqueeze(dim=1) # (N, 1, 1024, 1024)
        
        
        if self.MorSP_module:
            v = outputs[:, 1, :, :].unsqueeze(dim=1) # (N, 1, 1024, 1024)
            o = self.MorSP_module(o, torch.sigmoid(v))
        else: v = o
        
        loss = {}
        loss['loss_mask'] = self.loss_mask(o, batch_input["gt_mask"])
        loss['loss_skel'] = self.loss_skel(v, batch_input["gt_mask"])
        loss['total'] =  loss['loss_mask'] + loss['loss_skel']
        
        return loss
    
    def forward_eval(self, batch_input):
        input_image, sparse_embeddings, dense_embeddings, scale_size, original_size = self.get_input(batch_input)
        
        with torch.no_grad():
            # 图像编码
            image_embedding = self.image_encoder(input_image)  # (N, 256, 64, 64)
            # 解码获得低分辨率掩码
            low_res_masks, _ = self.mask_decoder(
                image_embeddings=image_embedding,  # (N, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (0, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (0, 256, 64, 64)
                multimask_output=True,
            )
            # 上采样
            outputs = self.postprocess_masks(low_res_masks,
                                            scale_size,
                                            original_size) # (N, 3, 1024, 1024)
            del image_embedding, low_res_masks # 删除无用变量
            # 骨架STD处理特征
            o = outputs[:, 0, :, :].unsqueeze(dim=1) # (N, 1, 1024, 1024)
        
        if self.MorSP_module:
            v = outputs[:, 1, :, :].unsqueeze(dim=1) # (N, 1, 1024, 1024)
            o = self.MorSP_module(o, torch.sigmoid(v))
        else: v = o
        
        return {"pred_mask": o,
                "skel_feat": v}
      
    def forward(self, batch_input, mode='train'):
        
        if mode == 'train':
            return self.forward_train(batch_input)
        elif mode == 'eval':
            return self.forward_eval(batch_input)
        else:
            raise ValueError(f"Invalid mode: {mode}. Please use 'train' or 'eval'.")
            