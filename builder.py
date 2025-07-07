import os
import logging
from functools import partial
from typing import Dict
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam, AdamW, SGD, RMSprop, LBFGS
from torch.nn import BCEWithLogitsLoss, MSELoss, BCELoss, SmoothL1Loss

from core.registry import get_loss_class, get_model_class, get_dataset_class

def build_loss(cfg: Dict):
    cfg_copy = cfg.copy()  
    loss_type = cfg_copy.pop('type')
    if loss_type in ['BCEWithLogitsLoss', 'MSELoss', 'BCELoss', 'SmoothL1Loss']:
        loss_class = eval(loss_type)
        if 'loss_weight' in cfg_copy:
            logging.info(f"torch.nn.{loss_type} doesn't support loss_weight")
            cfg_copy.pop('loss_weight')
    else:
        loss_class = get_loss_class(loss_type)
    return loss_class(**cfg_copy)


def build_model(cfg: Dict):
    cfg_copy = cfg.copy()
    model_type = cfg_copy.pop('type')
    model_class = get_model_class(model_type)
    if 'checkpoint' in cfg_copy:
        ckpt = cfg_copy.pop('checkpoint')
    else: ckpt = None

    logging.info(f'------- Building model: {model_type} --------')
    logging.info(f'Model config: {cfg_copy}')
    model = model_class(**cfg_copy)
    if ckpt:
        logging.info(f'Loading checkpoint from {ckpt}')
        load_ckpt(model, ckpt)
    
    return model


def load_ckpt(model: torch.nn.Module, ckpt: str):
    """
    Load the checkpoint file of a pre-trained model.

    Parameters:
    model (torch.nn.Module): The model instance to load the weights into.
    ckpt (str): The path to the pre-trained model checkpoint file.

    Returns:
    torch.nn.Module: The model instance with pre-trained weights loaded.
    """
    # Open the checkpoint file and load its contents
    assert os.path.exists(ckpt), f"Checkpoint file {ckpt} not found."
    with open(ckpt, "rb") as f:
        state_dict = torch.load(f, map_location='cpu')
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    # get state_dict for this model
    model_dict = model.state_dict()
    fliter_state_dict = {}
    
    # Iterate over the weights in the checkpoint
    for k, x in model_dict.items():
        if k in state_dict:
            # Check if the shape of the weight matches the shape of the current model's weight
            if x.shape == state_dict[k].shape:
                fliter_state_dict[k] = state_dict[k]
            else:
                logging.warning(f"Skip loading parameter: {k}, "
                                f"required shape: {x.shape}, "
                                f"loaded shape: {state_dict[k].shape}")
    
    # Update the current model's weight dictionary with the filtered weights
    logging.info(f"Loaded {len(fliter_state_dict)} / {len(model_dict)} parameters from pretrained weight file: {ckpt}")
    model_dict.update(fliter_state_dict)
    model.load_state_dict(model_dict)
    
    return model


def build_dataset(cfg: Dict):
    cfg_copy = cfg.copy()
    dataset_type = cfg_copy.pop('type')
    dataset_class = get_dataset_class(dataset_type)
    return dataset_class(**cfg_copy)


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def collate_fn(batch):
    # 过滤掉None的样本
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        # 如果整个batch都被过滤了，返回一个空batch
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def build_dataloader(dataset,
                     batch_size=1,
                     num_workers=4,
                     shuffle=True,
                     seed=1234,
                     distributed=False,
                     persistent_workers=False,
                     **kwargs):
    rank = 0
    world_size = 1
    assert batch_size % world_size == 0, \
        f"batch_size {batch_size} should be divisible by world_size {world_size}"
        
    if distributed:
        # 分布式训练设置
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed
        )
        batch_size = batch_size // world_size
    else:
        sampler = None

    init_fn = partial(
        worker_init_fn, 
        num_workers=num_workers, 
        rank=rank,
        seed=seed
    ) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=kwargs.pop('pin_memory', False),
        worker_init_fn=init_fn,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=kwargs.pop('prefetch_factor', 2),  # Fixed typo here (prefech_factor -> prefetch_factor)
        **kwargs
    )

    return data_loader


def build_optimizer(model, cfg):
    cfg_copy = cfg.copy()
    optimizer_type = cfg_copy.pop('type')
    if optimizer_type not in ['Adam', 'AdamW', 'SGD', 'RMSprop', 'LBFGS']:
        raise ValueError(f'unsupported optimizer type: {optimizer_type}')
    optimizer_class = eval(optimizer_type)
    optimizer = optimizer_class(filter(lambda p: p.requires_grad, model.parameters()), **cfg_copy)
    return optimizer

