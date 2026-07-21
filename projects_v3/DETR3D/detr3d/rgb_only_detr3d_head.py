"""A calibration-free DETR head for ScanNet axis-aligned boxes."""

import copy
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.structures import DepthInstance3DBoxes

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def make_mlp(channels: Sequence[int]) -> nn.Sequential:
    layers = []
    for index in range(len(channels) - 1):
        layers.append(nn.Linear(channels[index], channels[index + 1]))
        if index + 2 < len(channels):
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


@MODELS.register_module()
class RGBOnlyDETR3DHead(BaseModule):
    """Decode multi-view RGB tokens into 6D ScanNet boxes without calibration.

    ``forward`` accepts only FPN image features. Predicted boxes use the fixed
    ScanNet aligned range ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
    """

    def __init__(self,
                 num_classes: int = 18,
                 in_channels: int = 256,
                 embed_dims: int = 256,
                 num_queries: int = 900,
                 num_decoder_layers: int = 6,
                 num_heads: int = 8,
                 feedforward_channels: int = 1024,
                 max_views: int = 16,
                 pooled_size: int = 8,
                 center_range: Sequence[float] = (-6., -10., -1., 6., 10., 3.5),
                 size_range: Sequence[float] = (0.01, 0.01, 0.01, 8., 12., 4.),
                 anchor_grid_size: Sequence[int] = (15, 15, 4),
                 anchor_center_range: Sequence[float] = (-3.1, -4.1, .1, 3.1, 4.1, 1.95),
                 center_offset_scale: Sequence[float] = (3., 6., 2.),
                 size_prior: Sequence[float] = (.619, .632, .820),
                 max_detections: int = 100,
                 score_threshold: float = 0.05,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 loss_cls_weight: float = 2.0,
                 loss_bbox_weight: float = 5.0,
                 match_cls_weight: float = 2.0,
                 match_bbox_weight: float = 5.0,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        if len(center_range) != 6 or len(size_range) != 6:
            raise ValueError('center_range and size_range must each have six values.')
        if len(anchor_grid_size) != 3 or len(anchor_center_range) != 6:
            raise ValueError('anchor_grid_size and anchor_center_range must have three and six values.')
        if len(center_offset_scale) != 3 or len(size_prior) != 3:
            raise ValueError('center_offset_scale and size_prior must each have three values.')
        if int(anchor_grid_size[0]) * int(anchor_grid_size[1]) * int(anchor_grid_size[2]) != num_queries:
            raise ValueError('num_queries must equal the product of anchor_grid_size.')
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.max_views = max_views
        self.pooled_size = pooled_size
        self.max_detections = max_detections
        self.score_threshold = score_threshold
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.match_cls_weight = match_cls_weight
        self.match_bbox_weight = match_bbox_weight
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.register_buffer('center_min', torch.tensor(center_range[:3], dtype=torch.float32))
        self.register_buffer('center_max', torch.tensor(center_range[3:], dtype=torch.float32))
        self.register_buffer('size_min', torch.tensor(size_range[:3], dtype=torch.float32))
        self.register_buffer('size_max', torch.tensor(size_range[3:], dtype=torch.float32))
        self.register_buffer('center_offset_scale', torch.tensor(center_offset_scale, dtype=torch.float32))
        self.register_buffer('size_prior', torch.tensor(size_prior, dtype=torch.float32))
        anchor_axes = [
            torch.linspace(anchor_center_range[axis], anchor_center_range[axis + 3], int(anchor_grid_size[axis]))
            for axis in range(3)
        ]
        anchor_mesh = torch.meshgrid(*anchor_axes, indexing='ij')
        self.register_buffer('anchor_centers', torch.stack(anchor_mesh, dim=-1).reshape(-1, 3))

        self.input_proj = nn.Conv2d(in_channels, embed_dims, kernel_size=1)
        self.view_embedding = nn.Parameter(torch.zeros(max_views, embed_dims))
        self.token_embedding = nn.Parameter(torch.zeros(pooled_size * pooled_size, embed_dims))
        self.query_embedding = nn.Embedding(num_queries, embed_dims)
        self.anchor_position_embedding = make_mlp((3, embed_dims, embed_dims))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dims, nhead=num_heads,
            dim_feedforward=feedforward_channels, dropout=0.1,
            batch_first=True, norm_first=True)
        self.decoder = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        self.cls_branches = nn.ModuleList(
            [make_mlp((embed_dims, embed_dims, num_classes)) for _ in self.decoder])
        self.box_branches = nn.ModuleList(
            [make_mlp((embed_dims, embed_dims, 6)) for _ in self.decoder])
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.view_embedding, std=0.02)
        nn.init.normal_(self.token_embedding, std=0.02)
        nn.init.normal_(self.query_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        normalized_size_prior = ((self.size_prior - self.size_min) /
                                 (self.size_max - self.size_min)).clamp(.01, .99)
        size_prior_logits = torch.logit(normalized_size_prior)
        for branch in self.box_branches:
            last_layer = branch[-1]
            nn.init.constant_(last_layer.weight, 0)
            nn.init.constant_(last_layer.bias, 0)
            last_layer.bias.data[3:] = size_prior_logits

    def _build_memory(self, mlvl_feats: List[torch.Tensor]) -> torch.Tensor:
        feature = mlvl_feats[0]
        batch_size, num_views, channels, height, width = feature.shape
        if num_views > self.max_views:
            raise ValueError(f'Received {num_views} views but max_views={self.max_views}.')
        feature = feature.reshape(batch_size * num_views, channels, height, width)
        feature = self.input_proj(feature)
        feature = F.adaptive_avg_pool2d(feature, (self.pooled_size, self.pooled_size))
        feature = feature.flatten(2).transpose(1, 2)
        feature = feature.reshape(batch_size, num_views * self.pooled_size ** 2, self.embed_dims)
        token_position = self.token_embedding.repeat(num_views, 1)
        view_position = self.view_embedding[:num_views].repeat_interleave(
            self.pooled_size ** 2, dim=0)
        return feature + token_position.unsqueeze(0) + view_position.unsqueeze(0)

    def _decode_boxes(self, box_predictions: torch.Tensor, anchor_centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode anchor-relative center offsets and normalized dimensions."""
        center_offsets = torch.tanh(box_predictions[..., :3]) * self.center_offset_scale
        centers = anchor_centers + center_offsets
        centers = torch.maximum(torch.minimum(centers, self.center_max), self.center_min)
        sizes = self.size_min + box_predictions[..., 3:].sigmoid() * (self.size_max - self.size_min)
        return torch.cat((centers, sizes), dim=-1), center_offsets

    def _normalize_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        centers = (boxes[..., :3] - self.center_min) / (self.center_max - self.center_min)
        sizes = (boxes[..., 3:] - self.size_min) / (self.size_max - self.size_min)
        return torch.cat((centers, sizes), dim=-1).clamp(0., 1.)

    def forward(self, mlvl_feats: List[torch.Tensor],
                anchor_centers: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        memory = self._build_memory(mlvl_feats)
        if anchor_centers is None:
            anchor_centers = self.anchor_centers.unsqueeze(0).expand(memory.size(0), -1, -1)
        if anchor_centers.shape != (memory.size(0), self.query_embedding.num_embeddings, 3):
            raise ValueError(f'Expected anchor_centers [B, {self.query_embedding.num_embeddings}, 3], got {anchor_centers.shape}.')
        normalized_anchors = ((anchor_centers - self.center_min) /
                              (self.center_max - self.center_min)).clamp(0., 1.)
        query_positions = self.anchor_position_embedding(normalized_anchors)
        query = self.query_embedding.weight.unsqueeze(0) + query_positions
        
        all_cls_scores, all_bbox_preds, all_center_offsets = [], [], []
        for level, decoder in enumerate(self.decoder):
            query = decoder(query, memory)
            all_cls_scores.append(self.cls_branches[level](query))
            boxes, center_offsets = self._decode_boxes(self.box_branches[level](query), anchor_centers)
            all_bbox_preds.append(boxes)
            all_center_offsets.append(center_offsets)
        return {
            'all_cls_scores': torch.stack(all_cls_scores),
            'all_bbox_preds': torch.stack(all_bbox_preds),
            'all_center_offsets': torch.stack(all_center_offsets),
        }

    def _match(self, cls_logits: torch.Tensor, boxes: torch.Tensor,
               gt_boxes: torch.Tensor, gt_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if linear_sum_assignment is None:
            raise ImportError('RGBOnlyDETR3DHead requires scipy for Hungarian matching.')
        if len(gt_boxes) == 0:
            empty = torch.empty(0, dtype=torch.long, device=boxes.device)
            return empty, empty
        cls_cost = -cls_logits.sigmoid()[:, gt_labels]
        box_cost = torch.cdist(self._normalize_boxes(boxes), self._normalize_boxes(gt_boxes), p=1)
        query_indices, target_indices = linear_sum_assignment(
            (self.match_cls_weight * cls_cost + self.match_bbox_weight * box_cost).detach().cpu())
        return (torch.as_tensor(query_indices, device=boxes.device, dtype=torch.long),
                torch.as_tensor(target_indices, device=boxes.device, dtype=torch.long))

    def _loss_single(self, cls_logits: torch.Tensor, boxes: torch.Tensor,
                     batch_gt_instances_3d: List[InstanceData]) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_target = torch.zeros_like(cls_logits)
        matched_boxes, matched_targets = [], []
        for batch_index, instances in enumerate(batch_gt_instances_3d):
            gt_boxes = instances.bboxes_3d.tensor[:, :6].to(boxes.device)
            gt_labels = instances.labels_3d.to(boxes.device)
            query_indices, target_indices = self._match(
                cls_logits[batch_index], boxes[batch_index], gt_boxes, gt_labels)
            if len(query_indices):
                cls_target[batch_index, query_indices, gt_labels[target_indices]] = 1.
                matched_boxes.append(boxes[batch_index, query_indices])
                matched_targets.append(gt_boxes[target_indices])
        cross_entropy = F.binary_cross_entropy_with_logits(cls_logits, cls_target, reduction='none')
        probability = cls_logits.sigmoid()
        pt = probability * cls_target + (1. - probability) * (1. - cls_target)
        alpha = self.focal_alpha * cls_target + (1. - self.focal_alpha) * (1. - cls_target)
        loss_cls = (alpha * (1. - pt).pow(self.focal_gamma) * cross_entropy).mean()
        if matched_boxes:
            loss_bbox = F.l1_loss(
                self._normalize_boxes(torch.cat(matched_boxes)),
                self._normalize_boxes(torch.cat(matched_targets)), reduction='mean')
        else:
            loss_bbox = boxes.sum() * 0.
        return loss_cls * self.loss_cls_weight, loss_bbox * self.loss_bbox_weight

    def loss_by_feat(self, batch_gt_instances_3d: List[InstanceData],
                     preds_dicts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        cls_scores, bbox_preds = preds_dicts['all_cls_scores'], preds_dicts['all_bbox_preds']
        for level in range(cls_scores.size(0)):
            loss_cls, loss_bbox = self._loss_single(
                cls_scores[level], bbox_preds[level], batch_gt_instances_3d)
            prefix = '' if level == cls_scores.size(0) - 1 else f'd{level}.'
            losses[f'{prefix}loss_cls'] = loss_cls
            losses[f'{prefix}loss_bbox'] = loss_bbox
        return losses

    def predict_by_feat(self, preds_dicts: Dict[str, torch.Tensor]) -> List[InstanceData]:
        scores = preds_dicts['all_cls_scores'][-1].sigmoid()
        boxes = preds_dicts['all_bbox_preds'][-1]
        results = []
        for score, box in zip(scores, boxes):
            values, indices = score.flatten().topk(min(self.max_detections, score.numel()))
            labels = indices % self.num_classes
            query_indices = indices // self.num_classes
            keep = values >= self.score_threshold
            result = InstanceData()
            result.bboxes_3d = DepthInstance3DBoxes(
                box[query_indices[keep]], box_dim=6, with_yaw=False,
                origin=(0.5, 0.5, 0.5))
            result.scores_3d = values[keep]
            result.labels_3d = labels[keep]
            results.append(result)
        return results
