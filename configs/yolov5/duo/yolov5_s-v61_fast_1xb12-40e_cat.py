_base_ = '../yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

img_scale = (640, 640)
dataset_type = 'YOLOv5CocoDataset'
data_root = './data/cat/'
class_name = ('cat', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

# -----train val related-----
affine_scale = 0.5

anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]

max_epochs = 40
train_batch_size_per_gpu = 12
train_num_workers = 0
save_checkpoint_intervals = 10

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors)))

persistent_workers = False
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/trainval.json',
        data_prefix=dict(img='images/')))

# train_dataloader1 = dict(
#     batch_size=train_batch_size_per_gpu,
#     num_workers=train_num_workers,
#     persistent_workers=persistent_workers,
#     dataset=dict(
#         type=dataset_type))

# albu_train_transforms = [
#     dict(type='Blur', p=0.01),
#     dict(type='MedianBlur', p=0.01),
#     dict(type='ToGray', p=0.01),
#     dict(type='CLAHE', p=0.01)
# ]
#
# pre_transform = [
#     dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
#     dict(type='LoadAnnotations', with_bbox=True)
# ]

# train_pipeline1 = [
#     *pre_transform,
#     dict(
#         type='Mosaic',
#         img_scale=img_scale,
#         pad_val=114.0,
#         pre_transform=pre_transform),
#     dict(
#         type='YOLOv5RandomAffine',
#         max_rotate_degree=0.0,
#         max_shear_degree=0.0,
#         scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
#         # img_scale is (width, height)
#         border=(-img_scale[0] // 2, -img_scale[1] // 2),
#         border_val=(114, 114, 114)),
#     dict(
#         type='mmdet.Albu',
#         transforms=albu_train_transforms,
#         bbox_params=dict(
#             type='BboxParams',
#             format='pascal_voc',
#             label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
#         keymap={
#             'img': 'image',
#             'gt_bboxes': 'bboxes'
#         }),
#     dict(type='YOLOv5HSVRandomAug'),
#     dict(type='mmdet.RandomFlip', prob=0.5),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
#                    'flip_direction'))
# ]


# # train_dataloader1 = train_dataloader
# train_dataloader1 = dict(
#     batch_size=train_batch_size_per_gpu,
#     num_workers=train_num_workers,
#     persistent_workers=persistent_workers,
#     pin_memory=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='annotations/trainval.json',
#         data_prefix=dict(img='images/'),
#         filter_cfg=dict(filter_empty_gt=False, min_size=32),
#         pipeline=train_pipeline1),
#     collate_fn=dict(type='yolov5_collate'))


val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))
# train_cfg = dict(max_epochs=max_epochs, val_interval=10)
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa


train_cfg = dict(
    # type='EpochBasedTrainLoopWith2Loaders',
    max_epochs=max_epochs,
    val_interval=save_checkpoint_intervals)