_base_ = ["../../../_base_/default_runtime.py"]

# runtime
train_cfg = dict(max_epochs=210, val_interval=1)

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type="Adam",
        lr=5e-4,
    )
)

# learning policy
param_scheduler = [
    dict(
        type="LinearLR", begin=0, end=500, start_factor=0.001, by_epoch=False
    ),  # warm-up
    dict(
        type="MultiStepLR",
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True,
    ),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    checkpoint=dict(save_best="infinity/AP", rule="greater", max_keep_ckpts=2)
)

# codec settings
codec = dict(
    type="MSRAHeatmap",
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2,
    unbiased=True,
)

# model settings
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="HRNet",
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block="BOTTLENECK",
                num_blocks=(4,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block="BASIC",
                num_blocks=(4, 4),
                num_channels=(32, 64),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="BASIC",
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block="BASIC",
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
            ),
        ),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmpose/"
            "pretrain_models/hrnet_w32-36af842e.pth",
        ),
    ),
    head=dict(
        type="HeatmapHead",
        in_channels=32,
        out_channels=53,
        deconv_out_channels=None,
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode="heatmap",
        shift_heatmap=True,
    ),
)

# base dataset settings
dataset_type = "InfinityDataset"
data_mode = "topdown"
data_root = "../combined_dataset"

dataset_infinity = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file="train/annotations.json",
    data_prefix=dict(img="train/images/"),
    pipeline=[],
)
dataset_coco = dict(
    type="CocoDataset",
    data_root="../deep-high-resolution-net.pytorch/data/coco",
    data_mode=data_mode,
    ann_file="annotations/person_keypoints_val2017.json",
    data_prefix=dict(img="images/val2017/"),
    pipeline=[
        dict(
            type="KeypointConverter",
            num_keypoints=53,
            mapping=[
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (10, 10),
                (11, 11),
                (12, 12),
                (13, 13),
                (14, 14),
                (15, 15),
                (16, 16),
            ],
        )
    ],
)


# pipelines
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(type="RandomBBoxTransform"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

combined_dataset = dict(
    type="CombinedDataset",
    metainfo=dict(from_file="configs/_base_/datasets/infinity.py"),
    datasets=[dataset_infinity, dataset_coco],
    pipeline=train_pipeline,
    test_mode=False,
)

train_sampler = dict(
    type="MultiSourceSampler",
    batch_size=12,
    source_ratio=[1, 3],
    shuffle=True,
)

# data loaders
train_dataloader = dict(
    batch_size=12,
    num_workers=2,
    persistent_workers=True,
    # sampler=dict(type="DefaultSampler", shuffle=True),
    sampler=train_sampler,
    dataset=combined_dataset,
)
val_dataloader = dict(
    batch_size=6,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="test/annotations.json",
        data_prefix=dict(img="test/images/"),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type="InfinityMetric", ann_file=data_root + "/test/annotations.json", use_area=False
)
test_evaluator = val_evaluator

# visualizer
vis_backends = [
    dict(type="LocalVisBackend"),
    # dict(type='TensorboardVisBackend'),
    # dict(
    #     type="WandbVisBackend",
    #     init_kwargs=dict(
    #         project="synthetic_finetuning",
    #         entity="yonigoz",
    #         name="merge_infinity_coco/HRNet/w32_dark",
    #     ),
    # ),
]
visualizer = dict(
    type="PoseLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=10),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=10),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="PoseVisualizationHook", enable=True, interval=5),
)
