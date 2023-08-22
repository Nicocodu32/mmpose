# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from copy import deepcopy
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.fileio import dump, load
from mmengine.logging import MMLogger
from xtcocotools.cocoeval import COCOeval

from mmpose.registry import METRICS

from .coco_metric import CocoMetric


@METRICS.register_module()
class InfinityMetric(CocoMetric):
    """COCO pose estimation task evaluation metric.

    Evaluate AR, AP, and mAP for keypoint detection tasks. Support COCO
    dataset and other datasets in COCO format. Please refer to
    `COCO keypoint evaluation <https://cocodataset.org/#keypoints-eval>`__
    for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None
        use_area (bool): Whether to use ``'area'`` message in the annotations.
            If the ground truth annotations (e.g. CrowdPose, AIC) do not have
            the field ``'area'``, please set ``use_area=False``.
            Defaults to ``True``
        iou_type (str): The same parameter as `iouType` in
            :class:`xtcocotools.COCOeval`, which can be ``'keypoints'``, or
            ``'keypoints_crowd'`` (used in CrowdPose dataset).
            Defaults to ``'keypoints'``
        score_mode (str): The mode to score the prediction results which
            should be one of the following options:

                - ``'bbox'``: Take the score of bbox as the score of the
                    prediction results.
                - ``'bbox_keypoint'``: Use keypoint score to rescore the
                    prediction results.
                - ``'bbox_rle'``: Use rle_score to rescore the
                    prediction results.

            Defaults to ``'bbox_keypoint'`
        keypoint_score_thr (float): The threshold of keypoint score. The
            keypoints with score lower than it will not be included to
            rescore the prediction results. Valid only when ``score_mode`` is
            ``bbox_keypoint``. Defaults to ``0.2``
        nms_mode (str): The mode to perform Non-Maximum Suppression (NMS),
            which should be one of the following options:

                - ``'oks_nms'``: Use Object Keypoint Similarity (OKS) to
                    perform NMS.
                - ``'soft_oks_nms'``: Use Object Keypoint Similarity (OKS)
                    to perform soft NMS.
                - ``'none'``: Do not perform NMS. Typically for bottomup mode
                    output.

            Defaults to ``'oks_nms'`
        nms_thr (float): The Object Keypoint Similarity (OKS) threshold
            used in NMS when ``nms_mode`` is ``'oks_nms'`` or
            ``'soft_oks_nms'``. Will retain the prediction results with OKS
            lower than ``nms_thr``. Defaults to ``0.9``
        format_only (bool): Whether only format the output results without
            doing quantitative evaluation. This is designed for the need of
            test submission when the ground truth annotations are absent. If
            set to ``True``, ``outfile_prefix`` should specify the path to
            store the output results. Defaults to ``False``
        outfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., ``'a/b/prefix'``.
            If not specified, a temp file will be created. Defaults to ``None``
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Defaults to ``'cpu'``
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Defaults to ``None``
    """

    default_prefix: Optional[str] = "infinity"

    def __init__(
        self,
        ann_file: Optional[str] = None,
        use_area: bool = True,
        iou_type: str = "keypoints",
        score_mode: str = "bbox_keypoint",
        keypoint_score_thr: float = 0.2,
        nms_mode: str = "oks_nms",
        nms_thr: float = 0.9,
        format_only: bool = False,
        outfile_prefix: Optional[str] = None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(
            ann_file=ann_file,
            use_area=use_area,
            iou_type=iou_type,
            score_mode=score_mode,
            keypoint_score_thr=keypoint_score_thr,
            nms_mode=nms_mode,
            nms_thr=nms_thr,
            format_only=format_only,
            outfile_prefix=outfile_prefix,
            collect_device=collect_device,
            prefix=prefix,
        )
        self.infinity_keypoints_name = self.coco.loadCats(0)[0]["augmented_keypoints"]
        self.coco = None

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model, each of which has the following keys:

                - 'id': The id of the sample
                - 'img_id': The image_id of the sample
                - 'pred_instances': The prediction results of instance(s)
        """
        for data_sample in data_samples:
            if "pred_instances" not in data_sample:
                raise ValueError(
                    "`pred_instances` are required to process the "
                    f"predictions results in {self.__class__.__name__}. "
                )

            # keypoints.shape: [N, K, 2],
            # N: number of instances, K: number of keypoints
            # for topdown-style output, N is usually 1, while for
            # bottomup-style output, N is the number of instances in the image
            keypoints = data_sample["pred_instances"]["keypoints"]
            # [N, K], the scores for all keypoints of all instances
            keypoint_scores = data_sample["pred_instances"]["keypoint_scores"]
            assert keypoint_scores.shape == keypoints.shape[:2]

            # parse prediction results
            pred = dict()
            pred["id"] = data_sample["id"]
            pred["img_id"] = data_sample["img_id"]
            pred["keypoints"] = keypoints
            pred["keypoint_scores"] = keypoint_scores
            pred["category_id"] = data_sample.get("category_id", 1)

            if "bbox_scores" in data_sample["pred_instances"]:
                # some one-stage models will predict bboxes and scores
                # together with keypoints
                bbox_scores = data_sample["pred_instances"]["bbox_scores"]
            elif "bbox_scores" not in data_sample["gt_instances"] or len(
                data_sample["gt_instances"]["bbox_scores"]
            ) != len(keypoints):
                # bottom-up models might output different number of
                # instances from annotation
                bbox_scores = np.ones(len(keypoints))
            else:
                # top-down models use detected bboxes, the scores of which
                # are contained in the gt_instances
                bbox_scores = data_sample["gt_instances"]["bbox_scores"]
            pred["bbox_scores"] = bbox_scores

            # get area information
            if "bbox_scales" in data_sample["gt_instances"]:
                pred["areas"] = np.prod(
                    data_sample["gt_instances"]["bbox_scales"], axis=1
                )

            # parse gt
            gt = dict()

            if self.coco is None:
                gt["width"] = data_sample["ori_shape"][1]
                gt["height"] = data_sample["ori_shape"][0]
                gt["img_id"] = data_sample["img_id"]
                if self.iou_type == "keypoints_crowd":
                    assert "crowd_index" in data_sample, (
                        "`crowd_index` is required when `self.iou_type` is "
                        "`keypoints_crowd`"
                    )
                    gt["crowd_index"] = data_sample["crowd_index"]
                assert "raw_ann_info" in data_sample, (
                    "The row ground truth annotations are required for "
                    "evaluation when `ann_file` is not provided"
                )
                keypoints_list = deepcopy(data_sample["raw_ann_info"]["coco_keypoints"])
                for ipt, name in enumerate(self.infinity_keypoints_name):
                    keypoints_list += [
                        data_sample["raw_ann_info"]["keypoints"][name]["x"],
                        data_sample["raw_ann_info"]["keypoints"][name]["y"],
                        data_sample["raw_ann_info"]["keypoints"][name]["v"],
                    ]

                anns = deepcopy(data_sample["raw_ann_info"])
                anns["keypoints"] = keypoints_list
                gt["raw_ann_info"] = anns if isinstance(anns, list) else [anns]

            # add converted result to the results list
            self.results.append((pred, gt))


@METRICS.register_module()
class InfinityCocoMetric(CocoMetric):
    default_prefix: Optional[str] = "infinity_coco"

    def __init__(
        self,
        ann_file: Optional[str] = None,
        use_area: bool = True,
        iou_type: str = "keypoints",
        score_mode: str = "bbox_keypoint",
        keypoint_score_thr: float = 0.2,
        nms_mode: str = "oks_nms",
        nms_thr: float = 0.9,
        format_only: bool = False,
        outfile_prefix: Optional[str] = None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(
            ann_file=ann_file,
            use_area=use_area,
            iou_type=iou_type,
            score_mode=score_mode,
            keypoint_score_thr=keypoint_score_thr,
            nms_mode=nms_mode,
            nms_thr=nms_thr,
            format_only=format_only,
            outfile_prefix=outfile_prefix,
            collect_device=collect_device,
            prefix=prefix,
        )
        self.infinity_keypoints_name = self.coco.loadCats(0)[0]["augmented_keypoints"]
        self.coco = None

    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        """Do keypoint evaluation using COCOAPI.

        Args:
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            list: a list of tuples. Each tuple contains the evaluation stats
            name and corresponding stats value.
        """
        res_file = f"{outfile_prefix}.keypoints.json"
        coco_det = self.coco.loadRes(res_file)
        sigmas = self.dataset_meta["sigmas"][:17]
        coco_eval = COCOeval(self.coco, coco_det, self.iou_type, sigmas, self.use_area)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        if self.iou_type == "keypoints_crowd":
            stats_names = [
                "AP",
                "AP .5",
                "AP .75",
                "AR",
                "AR .5",
                "AR .75",
                "AP(E)",
                "AP(M)",
                "AP(H)",
            ]
        else:
            stats_names = [
                "AP",
                "AP .5",
                "AP .75",
                "AP (M)",
                "AP (L)",
                "AR",
                "AR .5",
                "AR .75",
                "AR (M)",
                "AR (L)",
            ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model, each of which has the following keys:

                - 'id': The id of the sample
                - 'img_id': The image_id of the sample
                - 'pred_instances': The prediction results of instance(s)
        """
        self.dataset_meta = deepcopy(self.dataset_meta)
        self.dataset_meta["num_keypoints"] = 17
        for data_sample in data_samples:
            if "pred_instances" not in data_sample:
                raise ValueError(
                    "`pred_instances` are required to process the "
                    f"predictions results in {self.__class__.__name__}. "
                )

            # keypoints.shape: [N, K, 2],
            # N: number of instances, K: number of keypoints
            # for topdown-style output, N is usually 1, while for
            # bottomup-style output, N is the number of instances in the image
            keypoints = data_sample["pred_instances"]["keypoints"]
            # [N, K], the scores for all keypoints of all instances
            keypoint_scores = data_sample["pred_instances"]["keypoint_scores"]
            assert keypoint_scores.shape == keypoints.shape[:2]

            # parse prediction results
            pred = dict()
            pred["id"] = data_sample["id"]
            pred["img_id"] = data_sample["img_id"]
            keypoints = keypoints[:, :17, :]
            pred["keypoints"] = keypoints
            keypoint_scores = keypoint_scores[:, :17]
            pred["keypoint_scores"] = keypoint_scores
            pred["category_id"] = data_sample.get("category_id", 1)

            if "bbox_scores" in data_sample["pred_instances"]:
                # some one-stage models will predict bboxes and scores
                # together with keypoints
                bbox_scores = data_sample["pred_instances"]["bbox_scores"]
            elif "bbox_scores" not in data_sample["gt_instances"] or len(
                data_sample["gt_instances"]["bbox_scores"]
            ) != len(keypoints):
                # bottom-up models might output different number of
                # instances from annotation
                bbox_scores = np.ones(len(keypoints))
            else:
                # top-down models use detected bboxes, the scores of which
                # are contained in the gt_instances
                bbox_scores = data_sample["gt_instances"]["bbox_scores"]
            pred["bbox_scores"] = bbox_scores

            # get area information
            if "bbox_scales" in data_sample["gt_instances"]:
                pred["areas"] = np.prod(
                    data_sample["gt_instances"]["bbox_scales"], axis=1
                )

            # parse gt
            gt = dict()

            if self.coco is None:
                gt["width"] = data_sample["ori_shape"][1]
                gt["height"] = data_sample["ori_shape"][0]
                gt["img_id"] = data_sample["img_id"]
                if self.iou_type == "keypoints_crowd":
                    assert "crowd_index" in data_sample, (
                        "`crowd_index` is required when `self.iou_type` is "
                        "`keypoints_crowd`"
                    )
                    gt["crowd_index"] = data_sample["crowd_index"]
                assert "raw_ann_info" in data_sample, (
                    "The row ground truth annotations are required for "
                    "evaluation when `ann_file` is not provided"
                )
                keypoints_list = deepcopy(data_sample["raw_ann_info"]["coco_keypoints"])
                anns = deepcopy(data_sample["raw_ann_info"])
                anns["keypoints"] = keypoints_list
                gt["raw_ann_info"] = anns if isinstance(anns, list) else [anns]

            # add converted result to the results list
            self.results.append((pred, gt))


@METRICS.register_module()
class InfinityAnatomicalMetric(CocoMetric):
    default_prefix: Optional[str] = "infinity_anatomical"

    def __init__(
        self,
        ann_file: Optional[str] = None,
        use_area: bool = True,
        iou_type: str = "keypoints",
        score_mode: str = "bbox_keypoint",
        keypoint_score_thr: float = 0.2,
        nms_mode: str = "oks_nms",
        nms_thr: float = 0.9,
        format_only: bool = False,
        outfile_prefix: Optional[str] = None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(
            ann_file=ann_file,
            use_area=use_area,
            iou_type=iou_type,
            score_mode=score_mode,
            keypoint_score_thr=keypoint_score_thr,
            nms_mode=nms_mode,
            nms_thr=nms_thr,
            format_only=format_only,
            outfile_prefix=outfile_prefix,
            collect_device=collect_device,
            prefix=prefix,
        )
        self.infinity_keypoints_name = self.coco.loadCats(0)[0]["augmented_keypoints"]
        self.coco = None

    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        """Do keypoint evaluation using COCOAPI.

        Args:
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            list: a list of tuples. Each tuple contains the evaluation stats
            name and corresponding stats value.
        """
        res_file = f"{outfile_prefix}.keypoints.json"
        coco_det = self.coco.loadRes(res_file)
        sigmas = self.dataset_meta["sigmas"][17:]
        coco_eval = COCOeval(self.coco, coco_det, self.iou_type, sigmas, self.use_area)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        if self.iou_type == "keypoints_crowd":
            stats_names = [
                "AP",
                "AP .5",
                "AP .75",
                "AR",
                "AR .5",
                "AR .75",
                "AP(E)",
                "AP(M)",
                "AP(H)",
            ]
        else:
            stats_names = [
                "AP",
                "AP .5",
                "AP .75",
                "AP (M)",
                "AP (L)",
                "AR",
                "AR .5",
                "AR .75",
                "AR (M)",
                "AR (L)",
            ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model, each of which has the following keys:

                - 'id': The id of the sample
                - 'img_id': The image_id of the sample
                - 'pred_instances': The prediction results of instance(s)
        """
        self.dataset_meta = deepcopy(self.dataset_meta)
        self.dataset_meta["num_keypoints"] = 36
        for data_sample in data_samples:
            if "pred_instances" not in data_sample:
                raise ValueError(
                    "`pred_instances` are required to process the "
                    f"predictions results in {self.__class__.__name__}. "
                )

            # keypoints.shape: [N, K, 2],
            # N: number of instances, K: number of keypoints
            # for topdown-style output, N is usually 1, while for
            # bottomup-style output, N is the number of instances in the image
            keypoints = data_sample["pred_instances"]["keypoints"]
            # [N, K], the scores for all keypoints of all instances
            keypoint_scores = data_sample["pred_instances"]["keypoint_scores"]
            assert keypoint_scores.shape == keypoints.shape[:2]

            # parse prediction results
            pred = dict()
            pred["id"] = data_sample["id"]
            pred["img_id"] = data_sample["img_id"]
            keypoints = keypoints[:, 17:, :]
            pred["keypoints"] = keypoints
            keypoint_scores = keypoint_scores[:, 17:]
            pred["keypoint_scores"] = keypoint_scores
            pred["category_id"] = data_sample.get("category_id", 0)

            if "bbox_scores" in data_sample["pred_instances"]:
                # some one-stage models will predict bboxes and scores
                # together with keypoints
                bbox_scores = data_sample["pred_instances"]["bbox_scores"]
            elif "bbox_scores" not in data_sample["gt_instances"] or len(
                data_sample["gt_instances"]["bbox_scores"]
            ) != len(keypoints):
                # bottom-up models might output different number of
                # instances from annotation
                bbox_scores = np.ones(len(keypoints))
            else:
                # top-down models use detected bboxes, the scores of which
                # are contained in the gt_instances
                bbox_scores = data_sample["gt_instances"]["bbox_scores"]
            pred["bbox_scores"] = bbox_scores

            # get area information
            if "bbox_scales" in data_sample["gt_instances"]:
                pred["areas"] = np.prod(
                    data_sample["gt_instances"]["bbox_scales"], axis=1
                )

            # parse gt
            gt = dict()

            if self.coco is None:
                gt["width"] = data_sample["ori_shape"][1]
                gt["height"] = data_sample["ori_shape"][0]
                gt["img_id"] = data_sample["img_id"]
                if self.iou_type == "keypoints_crowd":
                    assert "crowd_index" in data_sample, (
                        "`crowd_index` is required when `self.iou_type` is "
                        "`keypoints_crowd`"
                    )
                    gt["crowd_index"] = data_sample["crowd_index"]
                assert "raw_ann_info" in data_sample, (
                    "The row ground truth annotations are required for "
                    "evaluation when `ann_file` is not provided"
                )

                keypoints_list = []
                for ipt, name in enumerate(self.infinity_keypoints_name):
                    keypoints_list += [
                        data_sample["raw_ann_info"]["keypoints"][name]["x"],
                        data_sample["raw_ann_info"]["keypoints"][name]["y"],
                        data_sample["raw_ann_info"]["keypoints"][name]["v"],
                    ]
                anns = deepcopy(data_sample["raw_ann_info"])
                anns["keypoints"] = keypoints_list
                gt["raw_ann_info"] = anns if isinstance(anns, list) else [anns]

            # add converted result to the results list
            self.results.append((pred, gt))
