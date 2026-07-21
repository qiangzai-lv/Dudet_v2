"""Camera-free multi-view RGB DETR3D baseline for ScanNet."""

from typing import Dict, List

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample

from .detr3d import DETR3D


@MODELS.register_module()
class RGBOnlyDETR3D(DETR3D):
    """DETR3D detector that never constructs or consumes lidar2img.

    The parent class provides the multiview image backbone and FPN. The
    camera-dependent cross-attention head is replaced by RGBOnlyDETR3DHead.
    """

    def loss(self, batch_inputs_dict: Dict,
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> Dict:
        img_feats = self.extract_feat(
            batch_inputs_dict, [item.metainfo for item in batch_data_samples])
        predictions = self.pts_bbox_head(img_feats)
        return self.pts_bbox_head.loss_by_feat(
            [item.gt_instances_3d for item in batch_data_samples], predictions)

    def predict(self, batch_inputs_dict: Dict,
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        img_feats = self.extract_feat(
            batch_inputs_dict, [item.metainfo for item in batch_data_samples])
        predictions = self.pts_bbox_head(img_feats)
        results = self.pts_bbox_head.predict_by_feat(predictions)
        return self.add_pred_to_datasample(batch_data_samples, results)
