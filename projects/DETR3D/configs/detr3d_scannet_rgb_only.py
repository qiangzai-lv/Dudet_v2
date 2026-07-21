_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(imports=['projects.DETR3D.detr3d'], allow_failed_imports=False)
default_scope = 'mmdet3d'
find_unused_parameters = True

class_names = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain',
    'toilet', 'sink', 'bathtub', 'garbagebin'
]
metainfo = dict(classes=class_names)
data_root = '/root/damodel-tmp/data/ScanNet_processed/'
dataset_type = 'RGBOnlyScanNetDataset'
input_modality = dict(use_camera=True, use_lidar=False)

model = dict(
    type='RGBOnlyDETR3D',
    use_grid_mask=True,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=4),
    pts_bbox_head=dict(
        type='RGBOnlyDETR3DHead',
        num_classes=18,
        in_channels=256,
        embed_dims=256,
        num_queries=256,
        num_decoder_layers=6,
        num_heads=8,
        feedforward_channels=1024,
        max_views=16,
        pooled_size=8,
        # Fixed aligned ScanNet coordinate support for (cx, cy, cz, dx, dy, dz).
        center_range=[-6.0, -10.0, -1.0, 6.0, 10.0, 3.5],
        size_range=[0.01, 0.01, 0.01, 8.0, 12.0, 4.0],
        max_detections=100,
        score_threshold=0.05,
        loss_cls_weight=2.0,
        loss_bbox_weight=5.0,
        match_cls_weight=2.0,
        match_bbox_weight=5.0))

train_pipeline = [
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='SelectScanNetViews', num_views=16, random_select=True),
    dict(type='LoadRGBOnlyMultiViewImages', to_float32=True),
    dict(
        type='MultiViewWrapper',
        transforms=[dict(type='Resize', scale=(518, 392), keep_ratio=False)]),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'box_type_3d', 'num_views')),
]

test_pipeline = [
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='SelectScanNetViews', num_views=16, random_select=False),
    dict(type='LoadRGBOnlyMultiViewImages', to_float32=True),
    dict(
        type='MultiViewWrapper',
        transforms=[dict(type='Resize', scale=(518, 392), keep_ratio=False)]),
    dict(
        type='Pack3DDetInputs',
        keys=['img'],
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'box_type_3d', 'num_views')),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=6,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='scannet_infos_train_pts.pkl',
            pipeline=train_pipeline,
            modality=input_modality,
            metainfo=metainfo,
            filter_empty_gt=True,
            box_type_3d='Depth',
            test_mode=False)))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='scannet_infos_val_pts.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        filter_empty_gt=True,
        box_type_3d='Depth',
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(type='IndoorMetric')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-4),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 10, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=200,
         T_max=200, eta_min=1e-6),
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=5,
                    save_best='mAP_0.25', rule='greater'),
    logger=dict(type='LoggerHook', interval=20))
