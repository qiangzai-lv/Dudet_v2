"""Per-scene prediction export for the RGB-only ScanNet detector."""

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.utils import mkdir_or_exist

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class PerSceneResultDumpHook(Hook):
    """Save only prediction tensors required for offline visualization."""

    priority = 'LOW'

    def __init__(self, out_dir: str | None = None):
        self.out_dir = out_dir

    @staticmethod
    def _boxes_to_numpy(boxes: Any) -> np.ndarray:
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)
        centers = boxes.gravity_center
        return torch.cat((centers, boxes.tensor[:, 3:6]), dim=1).detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _scene_id(data_sample: Any, batch_idx: int, sample_index: int) -> str:
        paths = data_sample.metainfo.get('img_path', [])
        if isinstance(paths, str):
            paths = [paths]
        if paths:
            return Path(paths[0]).parent.name
        return f'batch_{batch_idx:05d}_{sample_index:02d}'

    def _destination(self, runner: Any) -> Path:
        output = Path(self.out_dir) if self.out_dir else Path(runner.work_dir) / 'scene_results'
        if not output.is_absolute():
            output = Path(runner.work_dir) / output
        mkdir_or_exist(output)
        return output

    def after_test_iter(self, runner: Any, batch_idx: int, data_batch: dict,
                        outputs: Sequence[Any]) -> None:
        output_dir = self._destination(runner)
        for sample_index, data_sample in enumerate(outputs):
            pred = data_sample.pred_instances_3d
            record: Dict[str, Any] = dict(
                format_version=2,
                scene_id=self._scene_id(data_sample, batch_idx, sample_index),
                pred_boxes=self._boxes_to_numpy(pred.bboxes_3d),
                pred_scores=pred.scores_3d.detach().cpu().numpy().astype(np.float32),
                pred_labels=pred.labels_3d.detach().cpu().numpy().astype(np.int64),
            )
            torch.save(record, output_dir / f"{record['scene_id']}.pt")
