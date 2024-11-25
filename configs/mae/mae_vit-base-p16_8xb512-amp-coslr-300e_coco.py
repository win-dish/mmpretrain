_base_ = [
    '../_base_/models/mae_vit-base-p16.py',
    #'../_base_/datasets/imagenet_bs512_mae.py',
    '../_base_/default_runtime.py',
]

# Override

data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CustomDataset',
        data_root='/mnt/c/Users/i0011180/Data/coco2017/',
        #C:\Users\i0011180\Data\coco2017\train2017
        #split='train',
        pipeline=train_pipeline))

#train_dataloader = dict(
#    dataset=dict(
#        type='CustomDataset',
#        data_root='/mnt/c/Users/i0011180/Data/coco2017/train2017',
#        split='train',
#        pieline=train_pipeline
#    )
#)

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-4 * 4096 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=260,
        by_epoch=True,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

randomness = dict(seed=0, diff_rank_seed=True)

# auto resume
resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096)
