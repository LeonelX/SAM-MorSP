task='train'
image_size = 512
epochs = 100
batchsize = 1

model = dict(
    type = "SAMMorSP",
    sam_type = "vit_b",
    sam_weights = "/home/xiej/data/models/sam_vit_b_01ec64.pth",
    lora_rank=4,
    MorSP = None,
    loss_mask=dict(
        type='BCEWithLogitsLoss'),
    loss_skel=dict(
        type="SoftCLDice",
        loss_weight=0.)
)

dataset_type = "SAMSegDataset"
data_root_dir = '/home/xiej/data/dataset'

data = dict(
    train=dict(
        type=dataset_type,
        ann_files=['data/WHU_train.json'],
        imgsz=image_size,
        transform=dict(
            crop_size=512,
            crop_ratio=0.,
            flip_h=0.5,
            flip_v=0.5)
        ),
    val=dict(
        type=dataset_type,
        ann_files=['data//WHU_val.json'],
        imgsz=image_size
        ),
    train_dataloader=dict(
        batch_size=batchsize,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=4,
        ),
    val_dataloader=dict(
        batch_size=batchsize,
        num_workers=8,
        shuffle=False,
        prefetch_factor=4,
        ),
    test=dict(
        type=dataset_type,
        ann_files=['data/WHU_test.json'],
        imgsz=image_size
    )
    )

# learning policy
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0,
    eps=1e-8,
    betas=(0.9, 0.999))

lr_config = dict(
    warm_up_epochs=5,
    cosine=True)
