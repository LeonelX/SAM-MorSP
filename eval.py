import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import json
import torch
import numpy as np 
import logging
from tqdm import tqdm

from models import *
from datasets import *
from builder import build_model, build_dataset, build_dataloader
from utils.process_config import load_config_as_dict
from utils.calculate_metric import evaluate_segement

def to_device(data, device):
    """Move tensors in a dictionary to device."""
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(element, device) for element in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def main(cfg_file, weight_path, device='cuda'):
    cfg = load_config_as_dict(cfg_file)
    sam_model = build_model(cfg['model']).cuda()
    sam_model.lora_sam.load_parameters(weight_path)
    sam_model.eval()

    test_set = build_dataset(cfg['data'].get('test', cfg['data']['val']))
    test_loader = build_dataloader(test_set, batch_size=4)
    total_metrices = dict()
    with tqdm(enumerate(test_loader), total=len(test_loader), leave=True) as pbar:
        for iter, batch in pbar:
            batch = to_device(batch, device)
            batch_output = sam_model(batch_input=batch, mode='eval')
            metrice = evaluate_segement(batch_output['pred_mask'], batch['gt_mask'])
            for k, v in metrice.items():
                total_metrices[k] = v + total_metrices.get(k, 0)
                
            pbar.set_postfix(Dice=total_metrices['DICE']/(iter+1), IoU=total_metrices['IOU']/(iter+1))
    
    for k, v in total_metrices.items():
        total_metrices[k] = v / len(test_loader)
    print(total_metrices)


if __name__ == '__main__':
    # 接受argparse传入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file', type=str, required=True)
    parser.add_argument('-w', '--weight_path', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args.cfg_file, args.weight_path, args.device)
