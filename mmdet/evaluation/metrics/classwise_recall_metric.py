# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric

from mmdet.evaluation.functional import bbox_overlaps
from mmdet.registry import METRICS


@METRICS.register_module()
class ClasswiseRecallMetric(BaseMetric):
    """Compute per-class and macro detection recall after detector NMS.

    Predictions are matched to ground-truth boxes class by class, in
    descending prediction-score order. Each ground-truth box can be matched
    once. IoU uses the legacy VOC coordinate convention to match
    :class:`VOCMetric`.

    Args:
        iou_thr (float): IoU threshold for a true positive. Defaults to 0.5.
        use_legacy_coordinate (bool): Use VOC's inclusive box coordinates.
            Defaults to True.
        collect_device (str): Device for distributed result collection.
            Defaults to 'cpu'.
        prefix (str, optional): Metric name prefix. Defaults to
            'exdark_recall'.
    """

    default_prefix: Optional[str] = 'exdark_recall'

    def __init__(self,
                 iou_thr: float = 0.5,
                 use_legacy_coordinate: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thr = float(iou_thr)
        self.use_legacy_coordinate = use_legacy_coordinate

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            gt_instances = self._get_field(sample, 'gt_instances')
            ignored_instances = self._get_field(sample, 'ignored_instances')
            pred_instances = self._get_field(sample, 'pred_instances')
            self.results.append({
                'gt_bboxes': self._to_numpy(
                    self._get_field(gt_instances, 'bboxes')),
                'gt_labels': self._to_numpy(
                    self._get_field(gt_instances, 'labels')),
                'ignored_bboxes': self._to_numpy(
                    self._get_field(ignored_instances, 'bboxes')),
                'ignored_labels': self._to_numpy(
                    self._get_field(ignored_instances, 'labels')),
                'pred_bboxes': self._to_numpy(
                    self._get_field(pred_instances, 'bboxes')),
                'pred_scores': self._to_numpy(
                    self._get_field(pred_instances, 'scores')),
                'pred_labels': self._to_numpy(
                    self._get_field(pred_instances, 'labels')),
            })

    @staticmethod
    def _get_field(container, name):
        if hasattr(container, name):
            return getattr(container, name)
        return container[name]

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, 'detach'):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def compute_metrics(self, results: list) -> dict:
        class_names = self.dataset_meta.get('classes')
        if not class_names:
            raise ValueError('Dataset metadata must provide class names.')

        num_classes = len(class_names)
        total_gt = np.zeros(num_classes, dtype=np.int64)
        total_tp = np.zeros(num_classes, dtype=np.int64)

        for result in results:
            gt_bboxes = np.asarray(result['gt_bboxes'], dtype=np.float32)
            gt_labels = np.asarray(result['gt_labels'], dtype=np.int64)
            ignored_bboxes = np.asarray(
                result['ignored_bboxes'], dtype=np.float32)
            ignored_labels = np.asarray(
                result['ignored_labels'], dtype=np.int64)
            pred_bboxes = np.asarray(result['pred_bboxes'], dtype=np.float32)
            pred_scores = np.asarray(result['pred_scores'], dtype=np.float32)
            pred_labels = np.asarray(result['pred_labels'], dtype=np.int64)

            for label in range(num_classes):
                gt_inds = np.flatnonzero(gt_labels == label)
                pred_inds = np.flatnonzero(pred_labels == label)
                class_gt = gt_bboxes[gt_inds]
                class_ignored = ignored_bboxes[ignored_labels == label]
                total_gt[label] += len(class_gt)
                all_class_gt = np.concatenate((class_gt, class_ignored), axis=0)
                if len(all_class_gt) == 0 or len(pred_inds) == 0:
                    continue

                class_pred = pred_bboxes[pred_inds]
                overlaps = bbox_overlaps(
                    class_pred,
                    all_class_gt,
                    use_legacy_coordinate=self.use_legacy_coordinate)
                order = np.argsort(-pred_scores[pred_inds], kind='mergesort')
                matched_gt = np.zeros(len(class_gt), dtype=bool)

                for pred_idx in order:
                    best_gt = int(np.argmax(overlaps[pred_idx]))
                    if overlaps[pred_idx, best_gt] < self.iou_thr:
                        continue
                    # VOC ignores detections whose best match is an ignored GT.
                    if best_gt >= len(class_gt):
                        continue
                    if not matched_gt[best_gt]:
                        matched_gt[best_gt] = True
                        total_tp[label] += 1

        class_recalls = np.divide(
            total_tp,
            total_gt,
            out=np.full(num_classes, np.nan, dtype=np.float64),
            where=total_gt > 0)
        iou_tag = f'{self.iou_thr:g}'

        metrics = {
            f'{name}_recall@{iou_tag}': float(np.round(class_recalls[i], 6))
            for i, name in enumerate(class_names)
            if total_gt[i] > 0
        }
        valid_recalls = class_recalls[total_gt > 0]
        if valid_recalls.size == 0:
            raise ValueError('No ground-truth boxes were found for recall.')
        metrics[f'macro_recall@{iou_tag}'] = float(
            np.round(valid_recalls.mean(), 6))
        metrics['num_classes'] = int(valid_recalls.size)
        return metrics
