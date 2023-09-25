_base_ = ['../../_base_/default_runtime.py', '../../_base_/det_p5_tta.py']

# ======================= Frequently modified parameters =====================
# -----data related-----
data_root = 'data/duo/'  # Root path of data
# Path of train annotation file
train_ann_file = 'annotations/instances_train.json'
train_data_prefix = 'images/train/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/instances_test.json'
val_data_prefix = 'images/test/'  # Prefix of val image path

class_name = ('holothurian', 'echinus', 'scallop', 'starfish')
num_classes = len(class_name)  # Number of classes for classification
metainfo = dict(classes=class_name,
                palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)])
# Batch size of a single GPU during training
train_batch_size_per_gpu = 16
train_batch_size_per_gpu_ml = 6
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8
# persistent_workers must be False if num_workers is 0
persistent_workers = False

# -----train val related-----
# Base learning rate for optim_wrapper
base_lr = 0.01
max_epochs = 400  # Maximum training epochs
num_last_epochs = 15  # Last epoch number to switch training pipeline
burnin_epoch = 200

# ======================= Possible modified parameters =======================
# -----data related-----
img_scale = (640, 640)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
lr_factor = 0.01  # Learning rate scaling factor
weight_decay = 0.0005
# Save model checkpoint and validation intervals
save_epoch_intervals = 10
val_intervals = 2
# The maximum checkpoints to keep.
max_keep_ckpts = 3
# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ============================== Unmodified in most cases ===================
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth'  # noqa

model = dict(
    type='EnYOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        # type='mmdet.DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    en_data_preprocessor=dict(
        type='EnDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv6EfficientRep',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        out_indices=(0, 1, 2, 3, 4),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True)),
    neck=dict(
        type='YOLOv6RepPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[128, 256, 512],
        num_csp_blocks=12,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    bbox_head=dict(
        type='YOLOv6Head',
        head_module=dict(
            type='YOLOv6HeadModule',
            num_classes=num_classes,
            in_channels=[128, 256, 512],
            widen_factor=widen_factor,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[8, 16, 32]),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='giou',
            bbox_format='xyxy',
            reduction='mean',
            loss_weight=2.5,
            return_iou=False)),
    en_head=dict(
        type='BaseEnHead',
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True)),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(
            type='BatchATSSAssigner',
            num_classes=num_classes,
            topk=9,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            topk=13,
            alpha=1,
            beta=6),
    ),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))

# The training pipeline of YOLOv6 is basically the same as YOLOv5.
# The difference is that Mosaic and RandomAffine will be closed in the last 15 epochs. # noqa
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True)
]

train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114),
        max_shear_degree=0.0),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_shear_degree=0.0,
    ),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    collate_fn=dict(type='yolov5_collate'),
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader

# =============================== Dataloader for enhancement ====================
syn_dataset_type = 'SynDataset'
syn_data_root = './data/synthesis'
pre_transform_enh = [
    dict(type='LoadSynImagesFromFile', backend_args=_base_.backend_args),
    # dict(type='LoadAnnotations', with_bbox=True)
]
train_pipeline_enh = [
    *pre_transform_enh,
    dict(type='ResizeSynImage',
        scale=img_scale, keep_ratio=True),
    dict(
        type='RandomFlipSynImage',
        prob=0.5),
    dict(type='PackEnInputs')
]
train_dataloader_enh = dict(
    batch_size=train_batch_size_per_gpu_ml,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=syn_dataset_type,
        pipeline=train_pipeline_enh,
        ann_file='train_infos.json',
        data_root=syn_data_root,
        data_prefix=dict(input='synthesis/', target='images/'),
        img_suffix='.png',
        test_mode=False),
    collate_fn=dict(type='synimage_collate'))

# =============================== Dataloader for detection in mutual-learning stage ====================
pre_transform_det = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True)
]
train_pipeline_det = [
    *pre_transform_det, # no mosaic_affine_pipeline and random colorazation
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='PPYOLOERandomCrop', aspect_ratio=[0.5, 1.0], thresholds=[0.2]),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader_det = dict(
    batch_size=train_batch_size_per_gpu_ml,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline_det)
)

# =============================== Dataloader for enhancement validation ==============================
val_enh_dataset_type= 'RwDataset'
val_enh_data_root = './data/real-world'
val_pipeline_enh = [
    dict(type='LoadRwImagesFromFile', backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=(512, 512)),
    dict(type='PackEnInputs', meta_keys=('file_name', 'img_shape', 'ori_shape',
                            'scale_factor'))
]

val_dataloader_enh = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=val_enh_dataset_type,
        pipeline=val_pipeline_enh,
        data_root=val_enh_data_root,
        img_suffix='.png'),
    collate_fn=dict(type='rwimage_collate')
)

val_enh_cfg = dict(type='EnhanceLoop')

# Optimizer and learning rate scheduler of YOLOv6 are basically the same as YOLOv5. # noqa
# The difference is that the scheduler_type of YOLOv6 is cosine.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=max_keep_ckpts,
        save_best='auto'))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - num_last_epochs,
        switch_pipeline=train_pipeline_stage2)
]

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop4EnYOLO',
    max_epochs=max_epochs,
    burnin_epoch= burnin_epoch,
    val_interval=val_intervals,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
