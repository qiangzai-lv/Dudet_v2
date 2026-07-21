"""ScanNet RGB-only dataset utilities for the DETR3D project."""

from os import path as osp
from typing import Callable, List, Optional, Union

import numpy as np
import mmcv
from mmengine.fileio import get
from mmcv import BaseTransform

from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS, TRANSFORMS
from mmdet3d.structures import DepthInstance3DBoxes


@DATASETS.register_module()
class RGBOnlyScanNetDataset(Det3DDataset):
    """ScanNet detection data that exposes RGB paths and GT boxes only."""

    METAINFO = {
        'classes': (
            'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
            'bookshelf', 'picture', 'counter', 'desk', 'curtain',
            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
            'garbagebin')
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_camera=True, use_lidar=False),
                 box_type_3d: str = 'Depth',
                 filter_empty_gt: bool = True,
                 remove_dontcare: bool = False,
                 test_mode: bool = False,
                 **kwargs) -> None:
        self.remove_dontcare = remove_dontcare
        super().__init__(
            data_root=data_root, ann_file=ann_file, metainfo=metainfo,
            pipeline=pipeline, modality=modality, box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt, test_mode=test_mode, **kwargs)
        if not self.modality.get('use_camera', False):
            raise ValueError('RGBOnlyScanNetDataset requires use_camera=True.')
        if self.modality.get('use_lidar', False):
            raise ValueError('RGBOnlyScanNetDataset does not support point-cloud input.')

    def parse_data_info(self, info: dict) -> dict:
        """Expose RGB paths without calibration, pose, or point-cloud fields."""
        image_paths = info.get('img_paths', [])
        if not image_paths:
            raise ValueError(f'Missing img_paths for sample {info.get("sample_idx", "<unknown>")}.')
        info = dict(info)
        info['img_path'] = [osp.join(self.data_root, path) for path in image_paths]
        for key in ('cam2img', 'lidar2cam', 'axis_align_matrix', 'lidar_points'):
            info.pop(key, None)
        if not self.test_mode:
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['ann_info'] = self.parse_ann_info(info)
            info['eval_ann_info'] = self._remove_dontcare(info['ann_info'])
        return info

    def parse_ann_info(self, info: dict) -> dict:
        ann_info = super().parse_ann_info(info)
        if self.remove_dontcare:
            ann_info = self._remove_dontcare(ann_info)
        if ann_info is None:
            ann_info = dict(
                gt_bboxes_3d=np.zeros((0, 6), dtype=np.float32),
                gt_labels_3d=np.zeros((0,), dtype=np.int64))
        ann_info['gt_bboxes_3d'] = DepthInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        for label in ann_info['gt_labels_3d']:
            if label != -1:
                self.num_ins_per_cat[label] += 1
        return ann_info


@TRANSFORMS.register_module()
class SelectScanNetViews(BaseTransform):
    """Select RGB frames using paths only; camera metadata is never loaded."""

    def __init__(self, num_views: int, random_select: bool = False) -> None:
        if num_views <= 0:
            raise ValueError('num_views must be positive.')
        self.num_views = num_views
        self.random_select = random_select

    def transform(self, results: dict) -> dict:
        paths = results['img_path']
        if len(paths) < self.num_views:
            raise ValueError(f'Only {len(paths)} views available, requested {self.num_views}.')
        if len(paths) == self.num_views:
            indices = np.arange(len(paths))
        elif self.random_select:
            indices = np.sort(np.random.choice(len(paths), self.num_views, replace=False))
        else:
            indices = np.linspace(0, len(paths) - 1, self.num_views, dtype=np.int64)
        results['img_path'] = [paths[index] for index in indices]
        results['num_views'] = self.num_views
        return results


@TRANSFORMS.register_module()
class LoadRGBOnlyMultiViewImages(BaseTransform):
    """Load multiple RGB images directly from ``img_path`` without calibration."""

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 backend_args: Optional[dict] = None) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.backend_args = backend_args

    def transform(self, results: dict) -> dict:
        filenames = results['img_path']
        image_bytes = [get(path, backend_args=self.backend_args) for path in filenames]
        images = [mmcv.imfrombytes(content, flag=self.color_type) for content in image_bytes]
        if any(image is None for image in images):
            missing = [path for path, image in zip(filenames, images) if image is None]
            raise ValueError(f'Unable to decode RGB images: {missing}')
        shapes = np.asarray([image.shape for image in images])
        max_shape = shapes.max(axis=0)
        if not np.all(shapes == max_shape):
            images = [mmcv.impad(image, shape=tuple(max_shape[:2]), pad_val=0)
                      for image in images]
        if self.to_float32:
            images = [image.astype(np.float32) for image in images]
        results['img'] = images
        results['filename'] = filenames
        results['img_shape'] = tuple(max_shape[:2])
        results['ori_shape'] = tuple(max_shape[:2])
        results['pad_shape'] = tuple(max_shape[:2])
        results['scale_factor'] = 1.0
        results['img_norm_cfg'] = dict(
            mean=np.zeros(3, dtype=np.float32),
            std=np.ones(3, dtype=np.float32),
            to_rgb=False)
        return results
