_base_ = [
    #'../_base_/datasets/imagenet_bs256_itpn.py',
    '../_base_/default_runtime.py',
]

# dataset settings
dataset_type = 'CustomDataset'
data_root = '/mnt/c/Users/i0011180/Data/coco2017'
data_preprocessor = dict(
    type='TwoNormDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # clip mean & std
    second_mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    second_std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomResizedCropAndInterpolationWithTwoPic',
        size=224,
        second_size=224,
        interpolation='bicubic',
        second_interpolation='bicubic',
        scale=(0.2, 1.0)),
    dict(
        type='BEiTMaskGenerator',
        input_size=(14, 14),
        num_masking_patches=75,
        max_num_patches=75,
        min_num_patches=16),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='',
        data_prefix=dict(img_path='train2017/'),
        pipeline=train_pipeline,
        with_label = False))


model = dict(
    type='iTPN',
    backbone=dict(
        type='iTPNHiViT',
        arch='base',
        drop_path_rate=0.0,
        rpe=True,
        layer_scale_init_value=0.1,
        reconstruction_type='clip'),
    neck=dict(
        type='iTPNPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        mlp_ratio=4.,
        reconstruction_type='clip',
        #  transformer pyramid
        fpn_dim=256,
        fpn_depth=2,
        num_outs=3,
    ),
    head=dict(
        type='iTPNClipHead',
        embed_dims=512,
        num_embed=512,
        loss=dict(type='CosineSimilarityLoss')),
    target_generator=dict(
        type='CLIPGenerator',
        tokenizer_path=  # noqa
        'https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/clip_vit_base_16.pth.tar'  # noqa
    ),
)

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    # betas: (0.9, 0.98) for 300 epochs and (0.9, 0.999) for 1600 epochs.
    optimizer=dict(
        type='AdamW', lr=1.5e-3, betas=(0.9, 0.98), weight_decay=0.05),
    clip_grad=dict(max_norm=3.0),
    paramwise_cfg=dict(
        custom_keys={
            '.norm': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            '.gamma': dict(decay_mult=0.0),
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        by_epoch=True,
        begin=10,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

randomness = dict(seed=0, diff_rank_seed=True)

find_unused_parameters = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)
