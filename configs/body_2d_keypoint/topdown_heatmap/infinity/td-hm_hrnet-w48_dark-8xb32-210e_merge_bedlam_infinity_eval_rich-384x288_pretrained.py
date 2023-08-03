_base_ = ["../../../_base_/default_runtime.py"]

# runtime
train_cfg = dict(max_epochs=30, val_interval=3)

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
        end=30,
        milestones=[14, 20],
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
    input_size=(288, 384),
    heatmap_size=(72, 96),
    sigma=3,
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
                num_channels=(48, 96),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="BASIC",
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block="BASIC",
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384),
            ),
        ),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="/scratch/users/yonigoz/mmpose_data/ckpts/hrnet/"
            "td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth",
            prefix="backbone",
        ),
    ),
    head=dict(
        type="HeatmapHead",
        in_channels=48,
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
data_root = "/scratch/users/yonigoz/infinity_datasets/"

dataset_infinity = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file="combined_dataset_15fps/train/annotations.json",
    data_prefix=dict(img=""),
    pipeline=[],
)

dataset_type = "InfinityDataset"
data_mode = "topdown"
data_root = "/scratch/users/yonigoz/BEDLAM/data/"

dataset_bedlam = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file="train_annotations.json",
    data_prefix=dict(img="training_images/"),
    pipeline=[],
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
    datasets=[dataset_infinity, dataset_bedlam],
    pipeline=train_pipeline,
    test_mode=False,
)

train_sampler = dict(
    type="MultiSourceSampler",
    batch_size=16,
    source_ratio=[1, 2],
    shuffle=True,
)

# data loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=train_sampler,
    dataset=combined_dataset,
)

dataset_type = "InfinityDataset"
data_mode = "topdown"
data_root = "/scratch/users/yonigoz/RICH/full_test/"


val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
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
            name="merge_bedlam_infinity_eval_rich/HRNet/w48_dark_pretrained",
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
    visualization=dict(type="PoseVisualizationHook", enable=True, interval=5),
)

work_dir = "/scratch/users/yonigoz/mmpose_data/work_dirs/merge_bedlam_infinity_eval_rich/HRNet/w48_dark_pretrained"
