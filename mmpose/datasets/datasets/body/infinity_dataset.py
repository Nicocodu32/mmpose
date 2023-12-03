import copy
import json
import os.path as osp
from copy import deepcopy
from itertools import filterfalse, groupby
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine import Config
from mmengine.dataset import BaseDataset, force_full_init
from mmengine.fileio import exists, get_local_path, load
from mmengine.utils import is_list_of
from scipy.io import loadmat
from xtcocotools.coco import COCO

from mmpose.registry import DATASETS
from mmpose.structures.bbox import bbox_cs2xyxy, bbox_xywh2xyxy

from ..base import BaseCocoStyleDataset
from ..utils import parse_pose_metainfo


@DATASETS.register_module(name="InfinityDataset")
class InfinityDataset(BaseCocoStyleDataset):
    METAINFO: dict = dict(from_file="configs/_base_/datasets/infinity.py")
    def __init__(self,
                 ann_file: str = '',
                 bbox_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 used_data_keys: Optional[Sequence[str]] = None):

        # select keypoints to be used in training
        self.used_data_keys = used_data_keys
        metainfo = self._check_metainfo(used_data_keys)

        super().__init__(
            ann_file=ann_file,
            bbox_file = bbox_file,
            data_mode = data_mode,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    @classmethod
    def _check_metainfo(cls, used_data_keys: Optional[Sequence[str]] = None):
        cfg_file = cls.METAINFO['from_file']
        metainfo = Config.fromfile(cfg_file).dataset_info
        keypoint_info = {}
        index = 0
        for _, keypoint in metainfo['keypoint_info'].items():
            name = keypoint['name']
            if used_data_keys is None or name in used_data_keys:
                keypoint['id'] = index
                keypoint_info[index] = keypoint
                index += 1
        metainfo['keypoint_info'] = keypoint_info

        return metainfo

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load data from annotations in COCO format."""

        assert exists(self.ann_file), "Annotation file does not exist"

        with get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)
        # set the metainfo about categories, which is a list of dict
        # and each dict contains the 'id', 'name', etc. about this category
        self.infinity_keypoints_name = self.coco.loadCats(0)[0]["augmented_keypoints"]
        self._metainfo["CLASSES"] = self.coco.loadCats(self.coco.getCatIds())

        instance_list = []
        image_list = []

        for img_id in self.coco.getImgIds():
            img = self.coco.loadImgs(img_id)[0]
            file_name = img["img_path"]
            img.update(
                {
                    "img_id": img_id,
                    "img_path": osp.join(self.data_prefix["img"], file_name),
                }
            )
            image_list.append(img)

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            for ann in self.coco.loadAnns(ann_ids):
                instance_info = self.parse_data_info(
                    dict(raw_ann_info=ann, raw_img_info=img)
                )

                # skip invalid instance annotation.
                if not instance_info:
                    continue

                instance_list.append(instance_info)
        return instance_list, image_list

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw COCO annotation of an instance.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict | None: Parsed instance annotation
        """

        ann = raw_data_info["raw_ann_info"]
        img = raw_data_info["raw_img_info"]

        # filter invalid instance
        if "bbox" not in ann or "keypoints" not in ann:
            return None

        img_w, img_h = img["width"], img["height"]

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann["bbox"]
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        keypoints_list = deepcopy(ann["coco_keypoints"])
        for ipt, name in enumerate(self.infinity_keypoints_name):
            if self.used_data_keys is None or name in self.used_data_keys:
                keypoints_list += [
                    ann["keypoints"][name]["x"],
                    ann["keypoints"][name]["y"],
                    ann["keypoints"][name]["v"],
                ]

        _keypoints = np.array(keypoints_list, dtype=np.float32).reshape(1, -1, 3)
        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2])

        if "num_keypoints" in ann:
            num_keypoints = ann["num_keypoints"]
        else:
            num_keypoints = np.count_nonzero(keypoints.max(axis=2))

        data_info = {
            "img_id": ann["image_id"],
            "img_path": img["img_path"],
            "bbox": bbox,
            "bbox_score": np.ones(1, dtype=np.float32),
            "num_keypoints": num_keypoints,
            "keypoints": keypoints,
            "keypoints_visible": keypoints_visible,
            "iscrowd": ann.get("iscrowd", 0),
            "segmentation": ann.get("segmentation", None),
            "id": ann["id"],
            "category_id": ann["category_id"],
            # store the raw annotation of the instance
            # it is useful for evaluation without providing ann_file
            "raw_ann_info": copy.deepcopy(ann),
        }

        if "crowdIndex" in img:
            data_info["crowd_index"] = img["crowdIndex"]

        return data_info
