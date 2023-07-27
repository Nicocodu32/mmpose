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
            checkpoint="/scratch/users/yonigoz/mmpose_data/ckpts/hrnet/"
            "td-hm_hrnet-w32_dark-8xb64-210e_coco-256x192-0e00bf12_20220914.pth",
            prefix="backbone",
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
data_root = "../"

# pipelines
train_pipeline = [
    dict(type="LoadImage", imdecode_backend="pillow"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(type="RandomBBoxTransform"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage", imdecode_backend="pillow"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="combined_dataset/train/annotations.json",
        data_prefix=dict(img=""),
        pipeline=train_pipeline,
    ),
)

dataset_type = "InfinityDataset"
data_mode = "topdown"
data_root = "/scratch/users/yonigoz/RICH/full_test/"

val_dataloader = dict(
    batch_size=32,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="val_annotations.json",
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(
        type="InfinityMetric",
        ann_file=data_root + "val_annotations.json",
        use_area=False,
    ),
    dict(
        type="InfinityCocoMetric",
        ann_file=data_root + "val_annotations.json",
        use_area=False,
    ),
    dict(
        type="InfinityAnatomicalMetric",
        ann_file=data_root + "val_annotations.json",
        use_area=False,
    ),
]

test_evaluator = val_evaluator

# visualizer
vis_backends = [
    dict(type="LocalVisBackend"),
    # dict(type='TensorboardVisBackend'),
    dict(
        type="WandbVisBackend",
        init_kwargs=dict(
            project="synthetic_finetuning",
            entity="yonigoz",
            name="infinity/HRNet/w32_dark_rich",
        ),
    ),
]
visualizer = dict(
    type="PoseLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=10),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(save_best="infinity/AP", rule="greater", max_keep_ckpts=2),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="PoseVisualizationHook", enable=True, interval=10),
)

work_dir = (
    "/scratch/users/yonigoz/mmpose_data/work_dirs/infinity_rich/HRNet/w32_dark_rich"
)
