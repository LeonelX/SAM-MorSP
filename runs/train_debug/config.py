# Auto-generated configuration file

task = 'train'

image_size = 512

epochs = 100

batchsize = 1

dataset_type = 'SAMSegDataset'

data_root_dir = '/home/xiej/data/dataset'

model = {
    'type': 'SAMMorSP',
    'sam_type': 'vit_b',
    'sam_weights': '/home/xiej/data/models/sam_vit_b_01ec64.pth',
    'lora_rank': 4,
    'MorSP': None,
    'loss_mask': {
        'type': 'BCEWithLogitsLoss'
    },
    'loss_skel': {
        'type': 'SoftCLDice',
        'loss_weight': 0.0
    }
}

data = {
    'train': {
        'type': 'SAMSegDataset',
        'ann_files': ['data/WHU_train.json'],
        'imgsz': 512,
        'transform': {
            'crop_size': 512,
            'crop_ratio': 0.0,
            'flip_h': 0.5,
            'flip_v': 0.5
        }
    },
    'val': {
        'type': 'SAMSegDataset',
        'ann_files': ['data//WHU_val.json'],
        'imgsz': 512
    },
    'train_dataloader': {
        'batch_size': 1,
        'num_workers': 8,
        'shuffle': True,
        'pin_memory': True,
        'prefetch_factor': 4
    },
    'val_dataloader': {
        'batch_size': 1,
        'num_workers': 8,
        'shuffle': False,
        'prefetch_factor': 4
    },
    'test': {
        'type': 'SAMSegDataset',
        'ann_files': ['data/WHU_test.json'],
        'imgsz': 512
    }
}

optimizer = {
    'type': 'AdamW',
    'lr': 0.0002,
    'weight_decay': 0,
    'eps': 1e-08,
    'betas': (0.9, 0.999)
}

