"""Point-cloud-derived spatial anchors for RGB-only ScanNet detection."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from mmcv import BaseTransform

from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.registry import MODELS, TRANSFORMS


@TRANSFORMS.register_module()
class LoadPointCloudAnchors(BaseTransform):
    """Convert an aligned ScanNet point cloud into fixed scene anchors.

    The raw point cloud is used only while preparing the sample. The detector
    receives the resulting anchor centers, not the point cloud itself.
    """

    def __init__(self,
                 num_anchors: int = 900,
                 voxel_size: float = 0.20,
                 max_candidates: int = 4096,
                 cache_dir: Optional[str] = None) -> None:
        if num_anchors <= 0 or voxel_size <= 0 or max_candidates < num_anchors:
            raise ValueError('Invalid point-anchor sampling parameters.')
        self.num_anchors = num_anchors
        self.voxel_size = voxel_size
        self.max_candidates = max_candidates
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _cache_path(self, point_path: Path) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        name = f'{point_path.stem}_anchors_{self.num_anchors}_v{self.voxel_size:.3f}.npy'
        return self.cache_dir / name

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        voxel_coords = np.floor(points / self.voxel_size).astype(np.int32)
        _, first_indices = np.unique(voxel_coords, axis=0, return_index=True)
        return points[np.sort(first_indices)]

    def _farthest_point_sample(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            raise ValueError('Point cloud has no valid points for anchor sampling.')
        if len(points) > self.max_candidates:
            indices = np.linspace(0, len(points) - 1, self.max_candidates, dtype=np.int64)
            points = points[indices]
        if len(points) < self.num_anchors:
            repeats = int(np.ceil(self.num_anchors / len(points)))
            return np.tile(points, (repeats, 1))[:self.num_anchors]

        selected = np.empty(self.num_anchors, dtype=np.int64)
        centroid = points.mean(axis=0, keepdims=True)
        selected[0] = np.argmin(((points - centroid) ** 2).sum(axis=1))
        distances = ((points - points[selected[0]]) ** 2).sum(axis=1)
        for index in range(1, self.num_anchors):
            selected[index] = np.argmax(distances)
            distances = np.minimum(
                distances, ((points - points[selected[index]]) ** 2).sum(axis=1))
        return points[selected]

    def _build_anchors(self, point_path: Path, axis_align: np.ndarray) -> np.ndarray:
        raw = np.fromfile(point_path, dtype=np.float32)
        if raw.size % 6 != 0:
            raise ValueError(f'Expected ScanNet [N, 6] point cloud, got {point_path}.')
        xyz = raw.reshape(-1, 6)[:, :3]
        homogeneous = np.concatenate(
            (xyz, np.ones((len(xyz), 1), dtype=np.float32)), axis=1)
        xyz = (homogeneous @ axis_align.T)[:, :3]
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        return self._farthest_point_sample(self._voxel_downsample(xyz)).astype(np.float32)

    def transform(self, results: dict) -> dict:
        point_path = Path(results['anchor_point_path'])
        cache_path = self._cache_path(point_path)
        anchors = None
        if cache_path is not None and cache_path.is_file():
            anchors = np.load(cache_path)
            if anchors.shape != (self.num_anchors, 3):
                anchors = None
        if anchors is None:
            anchors = self._build_anchors(
                point_path, np.asarray(results['anchor_axis_align_matrix'], dtype=np.float32))
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_name(f'{cache_path.name}.{os.getpid()}.tmp')
                with temporary.open('wb') as handle:
                    np.save(handle, anchors)
                os.replace(temporary, cache_path)
        results['anchor_centers'] = np.ascontiguousarray(anchors, dtype=np.float32)
        results.pop('anchor_point_path', None)
        results.pop('anchor_axis_align_matrix', None)
        return results


@TRANSFORMS.register_module()
class PackPointAnchorDetInputs(Pack3DDetInputs):
    """Pack images and point-cloud-derived anchor centers, but never raw points."""

    def pack_single_results(self, results: dict) -> dict:
        packed = super().pack_single_results(results)
        if 'anchor_centers' not in results:
            raise KeyError('LoadPointCloudAnchors must run before PackPointAnchorDetInputs.')
        packed['inputs']['anchor_centers'] = torch.as_tensor(
            results['anchor_centers'], dtype=torch.float32)
        return packed


@MODELS.register_module()
class PointAnchorDataPreprocessor(Det3DDataPreprocessor):
    """Preserve point-derived anchors while using the standard image pipeline."""

    def simple_process(self, data: dict, training: bool = False) -> dict:
        anchors = data['inputs'].get('anchor_centers')
        processed = super().simple_process(data, training)
        if anchors is None:
            raise KeyError('anchor_centers are required for point-anchor detection.')
        anchors = self.cast_data(anchors)
        if isinstance(anchors, (list, tuple)):
            anchors = torch.stack(list(anchors), dim=0)
        if anchors.ndim != 3 or anchors.shape[-1] != 3:
            raise ValueError(f'Expected anchor_centers [B, Q, 3], got {anchors.shape}.')
        processed['inputs']['anchor_centers'] = anchors
        return processed
