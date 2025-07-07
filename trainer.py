import os
import argparse
import logging
import math
import json

from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.process_config import load_config_as_dict, export_config
from utils.calculate_metric import evaluate_segement
from utils.dist_tools import gather_dict_metrics
from models import *
from datasets import *
from builder import build_model, build_dataset, build_dataloader, build_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument('--cfg_file', type=str, default='configs/morsp_cfg.py', help='config file path')
    parser.add_argument('--task_fold', type=str, default='runs/train_dubug', help='train task output path')
    # parser.add_argument('--init_method', type=str, help='Initialization method for distributed training')
    return parser.parse_args()

class Trainer:
    def __init__(self, cfg_file, args):
        
        self.cfg = load_config_as_dict(cfg_file)
        self.ddp = False
        self.init_distributed()
        
        if self.rank == 0:
            self.fold_path = args.task_fold
            os.makedirs(self.fold_path, exist_ok=True)
            self.setup_global_logging()
            self.logger.info("Loaded trainer config:\n%s", 
                        json.dumps(self.cfg, indent=4, ensure_ascii=False))
            export_config(self.cfg, os.path.join(self.fold_path, 'config.py'))
        
        self.set_model()
    
    def set_model(self, model=None):
        if model is not None and isinstance(model, torch.nn.Module):
            self.model = model
        else:
            self.model = build_model(self.cfg['model'])
        self.model = self.model.to(self.device)
        
        if self.rank == 0:
            self.logger.info(f'tunable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=False)
        
    def set_dataset(self):
        self.train_set = build_dataset(self.cfg['data']['train'])
        self.val_set = build_dataset(self.cfg['data']['val'])
    
        self.train_loader = build_dataloader(dataset=self.train_set, distributed=self.ddp,
                                             **self.cfg['data']['train_dataloader'])
        self.val_loader = build_dataloader(dataset=self.val_set, distributed=self.ddp,
                                           **self.cfg['data']['val_dataloader'])
        
    def setup_global_logging(self):
        """配置全局日志记录器"""
        log_file = os.path.join(self.fold_path, 'training.log')
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def init_distributed(self):
        # 检查是否由torchrun启动
        required_env_vars = {"RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
        if required_env_vars.issubset(os.environ.keys()):
            # 初始化分布式进程组
            dist.init_process_group(
                backend="nccl",  
                init_method="env://" 
            )
            self.rank = dist.get_rank()
            self.worldsize = dist.get_world_size()
        else:
            self.rank = 0
            self.worldsize = 1
            self.ddp = False
        
        if self.worldsize > 1: # 初始化分布式训练环境
            self.set_ddp()
        else: # 单GPU训练
            self.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        
    def set_ddp(self):
        self.ddp = True
        torch.cuda.set_device(self.rank)
        self.device = torch.device('cuda', self.rank)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
    
    def train(self):
        self.set_dataset()
        self.optimizer = build_optimizer(self.model, self.cfg['optimizer'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.Lr_lambda)
        self.scaler = torch.amp.GradScaler(self.device, enabled=True)
        
        self.epochs = self.cfg['epochs']
        train_metrics = []
        val_metrics = []
        best_score = 0
        
        for epoch in range(self.epochs):
            train_metric = self.train_epoch(epoch)
            val_metric = self.val_epoch(epoch)
            self.scheduler.step()
            
            if self.ddp:
                train_metric = gather_dict_metrics(train_metric, self.worldsize, self.device)
                val_metric = gather_dict_metrics(val_metric, self.worldsize, self.device)

            if self.rank == 0:
                score = 0.8 * val_metric['DICE'] + 0.2 * val_metric['IOU']
                
                self.logger.info('{:<10} {:<10} {:<10} {:<10}'.format(
                    'Epoch', 'train loss', 'DICE', 'IOU'))
                self.logger.info('{:<10} {:<10.4f} {:<10.4f} {:<10.4f}'.format(
                    str(epoch+1) + '/' + str(self.epochs), train_metric['total'], val_metric['DICE'], val_metric['IOU']))
                
                if score > best_score:
                    self.logger.info(f'save best model at epoch {epoch+1}')
                    self.save_ckpt(epoch+1, os.path.join(self.fold_path, 'best'))
                    best_score = score
                train_metrics.append(train_metric)
                val_metrics.append(val_metric)
                self.save_result(train_metrics, val_metrics)  
                self.save_ckpt(epoch+1, os.path.join(self.fold_path, 'last'))
        
        if self.rank == 0:
            self.logger.info(f'Training task finished')
    
    def train_epoch(self, epoch):
        self.model.train()
        total_losses = dict()
        valid_iter_count = 0
        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True) as pbar:
            for iter, batch in pbar:
                batch = self.to_device(batch, self.device)
                
                # scaler = torch.amp.GradScaler(self.device, enabled=True)
                if self.cfg.get("AMP", False):
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        losses = self.model(batch_input=batch, mode='train')
                else:
                    losses = self.model(batch_input=batch, mode='train')
                
                # 检查loss是否为nan
                if torch.isnan(losses['total']):
                    logging.warning(f'NaN loss detected at iteration {iter}, skipping update')
                    self.optimizer.zero_grad()  # 清空梯度
                    torch.cuda.empty_cache()  # 清空缓存
                    continue
                
                self.scaler.scale(losses['total']).backward()
                
                # 梯度累积策略（适合于显存不足的情况）
                if iter % self.cfg.get("batchsize_acc", 1) == 0 or iter == len(self.train_loader) - 1:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                for k, v in losses.items():
                    total_losses[k] = v.item() + total_losses.get(k, 0)
                
                valid_iter_count += 1
                pbar.set_postfix(Loss=total_losses['total']/(iter+1))
        
        for k, v in total_losses.items():
            total_losses[k] = v / valid_iter_count
        total_losses['lr'] = self.optimizer.param_groups[0]['lr']
        return total_losses
    
    def val_epoch(self, epoch):
        self.model.eval()
        total_metrices = dict()
        with tqdm(enumerate(self.val_loader), total=len(self.val_loader), leave=True) as pbar:
            for iter, batch in pbar:
                batch = self.to_device(batch, self.device)
                batch_output = self.model(batch_input=batch, mode='eval')
                metrice = evaluate_segement(batch_output['pred_mask'], batch['gt_mask'])
                for k, v in metrice.items():
                    total_metrices[k] = v + total_metrices.get(k, 0)
                    
                pbar.set_postfix(Dice=total_metrices['DICE']/(iter+1), IoU=total_metrices['IOU']/(iter+1))
        
        for k, v in total_metrices.items():
            total_metrices[k] = v / len(self.val_loader)
        
        return total_metrices     
    
    def save_ckpt(self, epoch, save_path):
        if self.rank == 0:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            model.lora_sam.save_parameters(save_path)
        else:
            logging.warning(f'Rank {self.rank} is not the main process, skipping checkpoint save.')
    
    
    def Lr_lambda(self, epoch, warm_up_steps=None, period=4):
        """学习率调整函数，用于LambdaLR调度器
        
        参数:
            epoch (int): 当前epoch数
            warm_up_steps (int, optional): 预热步数，默认为None
            period (int): 学习率调整周期，默认为4
            
        返回:
            float: 当前epoch对应的学习率缩放系数
        """
        if warm_up_steps is None:
            warm_up_steps = self.cfg["lr_config"].get("warm_up_epochs", 4)
        if epoch < warm_up_steps:
            # 预热阶段：学习率从0.8^(warm_up_steps)指数增长到1
            return 0.8 ** (warm_up_steps - epoch)
        else:
            if (epoch - warm_up_steps) < period:
                if self.cfg["lr_config"].get("cosine", False):
                    # 使用余弦退火调整学习率
                    return (1 + math.cos((epoch - warm_up_steps) * math.pi / period)) / 2
                else:
                    # 使用指数衰减调整学习率
                    return 0.95 ** (epoch - warm_up_steps)
            else:
                return self.Lr_lambda(epoch - warm_up_steps - period, warm_up_steps=0, period=period * 2)
                
    def to_device(self, data, device):
        """Move tensors in a dictionary to device."""
        if isinstance(data, dict):
            return {key: self.to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.to_device(element, device) for element in data]
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data
        
    def save_result(self, train_metrics, val_metrics):
        import pandas as pd
        colums = ['Epoch'] + list(train_metrics[0].keys()) + list(val_metrics[0].keys())
        metrics_df = pd.DataFrame(columns=colums)
        
        for epoch, (train_metric, val_metric) in enumerate(zip(train_metrics, val_metrics)):
            metrics_df.loc[epoch] = [epoch] + list(train_metric.values()) + list(val_metric.values())

        metrics_df.to_csv(os.path.join(self.fold_path, 'metrics.csv'), index=False)


def main():
    args = parse_args()
    trainer = Trainer(args.cfg_file, args)
    try:
        trainer.train()
    finally:
        if trainer.ddp:
            dist.destroy_process_group()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    