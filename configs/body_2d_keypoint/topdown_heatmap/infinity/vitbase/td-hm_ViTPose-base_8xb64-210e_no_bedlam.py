_base_ = ["../../../../_base_/default_runtime.py"]


used_data_keys=[
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "sternum",
        "rshoulder",
        "lshoulder",
        "r_lelbow",
        "l_lelbow",
        "r_melbow",
        "l_melbow",
        "r_lwrist",
        "l_lwrist",
        "r_mwrist",
        "l_mwrist",
        "r_ASIS",
        "l_ASIS",
        "r_PSIS",
        "l_PSIS",
        "r_knee",
        "l_knee",
        "r_mknee",
        "l_mknee",
        "r_ankle",
        "l_ankle",
        "r_mankle",
        "l_mankle",
        "r_5meta",
        "l_5meta",
        "r_toe",
        "l_toe",
        "r_big_toe",
        "l_big_toe",
        "l_calc",
        "r_calc",
        "C7",
        "L2",
        "T11",
        "T6",
    ]

# runtime
train_cfg = dict(max_epochs=30, val_interval=3)

# optimizer
custom_imports = dict(
    imports=["mmpose.engine.optim_wrappers.layer_decay_optim_wrapper"],
    allow_failed_imports=False,
)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        custom_keys={
            "bias": dict(decay_multi=0.0),
            "pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        },
    ),
    constructor="LayerDecayOptimWrapperConstructor",
    clip_grad=dict(max_norm=1.0, norm_type=2),
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
        milestones=[15, 20],
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
codec = dict(type="UDPHeatmap", input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

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
        type="mmcls.VisionTransformer",
        arch="base",
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cls_token=False,
        output_cls_token=False,
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="/scratch/users/yonigoz/mmpose_data/ckpts/vit/"
            "td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth",
            prefix="backbone",
        ),
    ),
    head=dict(
        type="HeatmapHead",
        in_channels=768,
        out_channels=len(used_data_keys),
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode="heatmap",
        shift_heatmap=False,
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
    used_data_keys=used_data_keys,
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
    used_data_keys=used_data_keys,
)

dataset_type = "InfinityDataset"
data_mode = "topdown"
data_root = "/scratch/users/yonigoz/"

dataset_3DPW = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file="3DPW/train_annotations.json",
    data_prefix=dict(img=""),
    pipeline=[],
    used_data_keys=used_data_keys,
)

dataset_type = "CocoDataset"
data_mode = "topdown"
data_root = "/scratch/users/yonigoz/coco_dataset"


dataset_coco = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file="annotations/person_keypoints_train2017.json",
    data_prefix=dict(img="images/train2017/"),
    pipeline=[
        dict(
            type="KeypointConverter",
            num_keypoints=len(used_data_keys),
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
    dict(type="TopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="PackPoseInputs"),
]

test_pipeline = val_pipeline

combined_dataset = dict(
    type="CombinedDataset",
    metainfo=dict(from_file="configs/_base_/datasets/infinity.py"),
    datasets=[dataset_infinity, dataset_3DPW, dataset_coco],
    pipeline=train_pipeline,
    used_data_keys=used_data_keys,
    test_mode=False,
)

train_sampler = dict(
    type="MultiSourceSampler",
    batch_size=32,
    source_ratio=[1, 1, 1],
    shuffle=True,
)

# data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=train_sampler,
    dataset=combined_dataset,
)


dataset_type = "InfinityDataset"
data_mode = "topdown"
data_root = ""
# data_root = "/scratch/users/yonigoz/BEDLAM/data/"

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="/scratch/users/yonigoz/RICH/downsampled/val_annotations.json",
        # data_prefix=dict(img="eval_images/"),
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=val_pipeline,
        used_data_keys=used_data_keys,
    ),
)
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(
        type="InfinityMetric",
        ann_file=data_root + "/scratch/users/yonigoz/RICH/downsampled/val_annotations.json",
        use_area=False,
        used_data_keys=used_data_keys,
    ),
    dict(
        type="InfinityCocoMetric",
        ann_file=data_root + "/scratch/users/yonigoz/RICH/downsampled/val_annotations.json",
        use_area=False,
        used_data_keys=used_data_keys,
    ),
    dict(
        type="InfinityAnatomicalMetric",
        ann_file=data_root + "/scratch/users/yonigoz/RICH/downsampled/val_annotations.json",
        use_area=False,
        used_data_keys=used_data_keys,
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
            name="no_bedlam/ViT/base_pretrained",
        ),
    ),
]
visualizer = dict(
    type="PoseLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=1000),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(save_best="infinity/AP", rule="greater", max_keep_ckpts=2),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="PoseVisualizationHook", enable=True, interval=20),
)

work_dir = "/scratch/users/yonigoz/mmpose_data/work_dirs/no_bedlam/ViT/base_pretrained"
