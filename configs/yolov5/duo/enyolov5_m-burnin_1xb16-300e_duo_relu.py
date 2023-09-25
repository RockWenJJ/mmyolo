_base_ = './yolov5_s-v61_syncbn-fast_8xb16-300e_duo.py'

# ========================Frequently modified parameters======================
# -----data related-----
data_root = './data/duo/'  # Root path of data
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

# -----model related-----
# Basic size of multi-scale prior box
anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)]  # P5/32
]

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs
base_lr = 0.01 #0.01
max_epochs = 300  # Maximum training epochs
burnin_epoch = 300

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.65),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

# ========================Possible modified parameters========================
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
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32,
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5)

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.67
# The scaling factor that controls the width of the network structure
widen_factor = 0.75
# Strides of multi-scale prior box
strides = [8, 16, 32]
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0
prior_match_thr = 4.  # Priori box matching threshold
# The obj loss weights of the three output layers
obj_level_weights = [4., 1., 0.4]
lr_factor = 0.01  # Learning rate scaling factor
weight_decay = 0.0005
# Save model checkpoint and validation intervals
save_checkpoint_intervals = 10
val_intervals = 2
# The maximum checkpoints to keep.
max_keep_ckpts = 3
# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth'   # noqa

model = dict(
    _delete_=True,
    type='EnYOLODetector',
    data_preprocessor=dict(
        # type='YOLOv5DetDataPreprocessor',
        type='mmdet.DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    en_data_preprocessor=dict(
        type='EnDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv5CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        out_indices=(0, 1, 2, 3, 4), # different from the original setting
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=widen_factor,
            featmap_strides=strides,
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=strides),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_cls_weight *
            (num_classes / num_classes * 3 / num_det_layers)),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-7,
            reduction='mean',
            loss_weight=loss_bbox_weight * (3 / num_det_layers),
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_obj_weight *
            ((img_scale[0] / 640)**2 * 3 / num_det_layers)),
        prior_match_thr=prior_match_thr,
        obj_level_weights=obj_level_weights),
    en_head=dict(
        type='BaseEnHead',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True)),
    test_cfg=model_test_cfg)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)))


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

# ================================ Dataloader for validation =========================================
val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix)))

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'annotations/instances_test.json')
test_evaluator = val_evaluator

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

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=50))

train_cfg = dict(
    type='EpochBasedTrainLoop4EnYOLO',
    max_epochs=max_epochs,
    burnin_epoch= burnin_epoch,
    val_interval=val_intervals)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(lr=base_lr))