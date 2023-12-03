# Copyright (c) OpenMMLab. All rights reserved.
import glob
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
import torch
from tqdm import tqdm

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

markerNames = [
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
        "r_bpinky",
        "l_bpinky",
        "r_tpinky",
        "l_tpinky",
        "r_bindex",
        "l_bindex",
        "r_tindex",
        "l_tindex",
        "r_tmiddle",
        "l_tmiddle",
        "r_tring",
        "l_tring",
        "r_bthumb",
        "l_bthumb",
        "r_tthumb",
        "l_tthumb",
        "C7",
        "L2",
        "T11",
        "T6",
    ]

markerPairs = {
        "rshoulder":"lshoulder",
        "lshoulder":"rshoulder",
        "r_lelbow":"l_lelbow",
        "l_lelbow":"r_lelbow",
        "r_melbow":"l_melbow",
        "l_melbow":"r_melbow",
        "r_lwrist":"l_lwrist",
        "l_lwrist":"r_lwrist",
        "r_mwrist":"l_mwrist",
        "l_mwrist":"r_mwrist",
        "r_ASIS":"l_ASIS",
        "l_ASIS":"r_ASIS",
        "r_PSIS":"l_PSIS",
        "l_PSIS":"r_PSIS",
        "r_knee":"l_knee",
        "l_knee":"r_knee",
        "r_mknee":"l_mknee",
        "l_mknee":"r_mknee",
        "r_ankle":"l_ankle",
        "l_ankle":"r_ankle",
        "r_mankle":"l_mankle",
        "l_mankle":"r_mankle",
        "r_5meta":"l_5meta",
        "l_5meta":"r_5meta",
        "r_toe":"l_toe",
        "l_toe":"r_toe",
        "r_big_toe":"l_big_toe",
        "l_big_toe":"r_big_toe",
        "l_calc":"r_calc",
        "r_calc":"l_calc",
        "r_bpinky":"l_bpinky",
        "l_bpinky":"r_bpinky",
        "r_tpinky":"l_tpinky",
        "l_tpinky":"r_tpinky",
        "r_bindex":"l_bindex",
        "l_bindex":"r_bindex",
        "r_tindex":"l_tindex",
        "l_tindex":"r_tindex",
        "r_tmiddle":"l_tmiddle",
        "l_tmiddle":"r_tmiddle",
        "r_tring":"l_tring",
        "l_tring":"r_tring",
        "r_bthumb":"l_bthumb",
        "l_bthumb":"r_bthumb",
        "r_tthumb":"l_tthumb",
        "l_tthumb":"r_tthumb",
    }

DET_CONFIG = "demo/mmdetection_cfg/configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py"
DET_CHECKPOINT = "https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth"

# DET_CONFIG = "demo/mmdetection_cfg/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco"
# DET_CHECKPOINT = "https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth"

# DET_CONFIG = "demo/mmdetection_cfg/mask_rcnn_r50_fpn_2x_coco.py"
# DET_CHECKPOINT = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"

# DET_CONFIG = "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
# DET_CHECKPOINT = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

# DET_CONFIG = "demo/mmdetection_cfg/yolov3_d53_320_273e_coco.py"
# DET_CHECKPOINT = "https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth"

POSE_CONFIG = "configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_hrnet-w48_dark-8xb32-210e_merge_bedlam_infinity_eval_bedlam-384x288_pretrained.py"
POSE_CHECKPOINT = "pretrain/hrnet/best_infinity_AP_epoch_21.pth"

# POSE_CONFIG = "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_dark-8xb64-210e_coco-256x192.py"
# POSE_CHECKPOINT = 'mmpose_data/ckpts/td-hm_hrnet-w32_dark-8xb64-210e_coco-256x192-0e00bf12_20220914.pth'

# # POSE_CONFIG = "configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_ViTPose-huge_8xb64-210e_merge_bedlam_infinity_eval_rich-256x192.py"
# # POSE_CHECKPOINT = "pretrain/vit/best_infinity_AP_epoch_4.pth"

OPENCAP_ROOT = "../OpenCap/data"
OUTPUT_ROOT = "eval_opencap_output_hrnet_21e"

BBOX_THR = 0.3
NMS_THR = 0.3
KPT_THR = 0.3


try:
    from mmdet.apis import inference_detector, init_detector

    print("success")
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    print("failed")


def process_one_image(img, detector, pose_estimator, visualizer=None, show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
    )
    bboxes = bboxes[
        np.logical_and(
            pred_instance.labels == 0,
            pred_instance.scores > BBOX_THR,
        )
    ]
    bboxes = bboxes[nms(bboxes, NMS_THR), :4]
    if len(bboxes) > 1:
        # only keep the largest bbox
        bboxes = bboxes[
            np.argmax((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]))
        ][None, :]

    # predict keypoints
    torch.cuda.empty_cache()
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order="rgb")
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)
    if visualizer is not None:
        visualizer.add_datasample(
            "result",
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=True,
            show_kpt_idx=False,
            skeleton_style="mmpose",
            show=False,
            wait_time=show_interval,
            kpt_thr=KPT_THR,
        )

    # if there is no instance detected, return None
    return data_samples.get("pred_instances", None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--det-config", default=DET_CONFIG, help="Config file for detection"
    )
    parser.add_argument(
        "--det-checkpoint", default=DET_CHECKPOINT, help="Checkpoint file for detection"
    )
    parser.add_argument(
        "--pose-config", default=POSE_CONFIG, help="Config file for pose"
    )
    parser.add_argument(
        "--pose-checkpoint", default=POSE_CHECKPOINT, help="Checkpoint file for pose"
    )

    parser.add_argument(
        "--opencap-root", default=OPENCAP_ROOT, help="Checkpoint file for pose"
    )
    parser.add_argument(
        "--output-root", default=OUTPUT_ROOT, help="Checkpoint file for pose"
    )

    parser.add_argument(
        "--save-videos",
        action="store_true",
        default=False,
        help="whether to save predicted results as videos",
    )

    args = parser.parse_args()

    # check if cuda is available to set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # build detector
    detector = init_detector(args.det_config, args.det_checkpoint, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
    )

    # build visualizer
    pose_estimator.cfg.visualizer.radius = 3
    pose_estimator.cfg.visualizer.alpha = 0.8
    pose_estimator.cfg.visualizer.line_width = 1
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style="mmpose")
    # get all files that end with _syncdWithMocap.avi
    video_synced_files = glob.glob(
        f"{args.opencap_root}/**/*_syncdWithMocap.avi", recursive=True
    )
    for video_file in tqdm(video_synced_files):
        print("video_file", video_file)
        if "subject8" not in video_file or "DJAsym3" not in video_file or "Cam2" not in video_file:
            continue
        output_file = os.path.join(
            args.output_root,
            video_file.replace(args.opencap_root, "")[1:],
        )
        pred_save_path = os.path.join(
            args.output_root,
            video_file.replace(args.opencap_root, "")[1:].replace(".avi", ".json"),
        )
        mmengine.mkdir_or_exist("/".join(output_file.split("/")[:-1]))
        print("output_file", output_file)
        cap = cv2.VideoCapture(video_file)

        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # topdown pose estimation
            pred_instances = process_one_image(
                frame, detector, pose_estimator, visualizer, 0.001
            )

            # save prediction results
            pred_instances_list.append(
                dict(frame_id=frame_idx, instances=split_instances(pred_instances))
            )

            # output videos
            if output_file and args.save_videos:
                # frame_vis = visualizer.get_image()
                for index_person in range(len(pred_instances["bboxes"])):
                    bbox = pred_instances["bboxes"][index_person]
                    kpts = pred_instances["keypoints"][index_person]
                    # cv2.rectangle(
                    #     frame,
                    #     (int(bbox[0]), int(bbox[1])),
                    #     (int(bbox[2]), int(bbox[3])),
                    #     (0, 255, 0),
                    #     2,
                    # )
                    for index_kpt in range(len(kpts)):



                        if index_kpt < 17:
                            continue
                            color = (0, 0, 255)

                        else:
                            # continue
                            if 31 <=index_kpt-17<=46 or markerNames[index_kpt-17] in ["l_calc","r_calc"]:
                                continue
                            if markerNames[index_kpt-17] in markerPairs.keys() :
                                if markerNames[index_kpt-17][0] == "l":
                                    color = (255, 0, 255)
                                else:
                                    color = (0 , 255, 255)
                            else:
                                color = (255, 0, 0)

                        cv2.circle(
                            frame,
                            (int(kpts[index_kpt][0]), int(kpts[index_kpt][1])),
                            3,
                            color,
                            -1,
                        )

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        25,  # saved fps
                        (frame.shape[1], frame.shape[0]),
                    )

                video_writer.write(mmcv.rgb2bgr(frame))

        if video_writer:
            video_writer.release()

        cap.release()

        with open(pred_save_path, "w") as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list,
                ),
                f,
                indent="\t",
            )
        print(f"predictions have been saved at {pred_save_path}")


if __name__ == "__main__":
    main()
