#!/usr/bin/env python3
"""Render one saved RGB-only DETR3D ScanNet prediction."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.fileio import load

CLASS_NAMES = (
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain',
    'toilet', 'sink', 'bathtub', 'garbagebin')
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))


def box_corners(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0, 8, 3), dtype=np.float32)
    offsets = np.array([
        [-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
        [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5],
    ], dtype=np.float32)
    return boxes[:, None, :3] + offsets[None] * boxes[:, None, 3:6]


def load_scene(scene_id: str, data_root: Path, ann_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    annotations = load(ann_file)
    data_list = annotations['data_list'] if isinstance(annotations, dict) else annotations
    info = next((
        item for item in data_list
        if (str(item.get('sample_idx')) == scene_id
            or any(Path(path).parent.name == scene_id for path in item.get('img_paths', []))
            or Path(item.get('lidar_points', {}).get('lidar_path', '')).stem == scene_id)
    ), None)
    if info is None:
        raise KeyError(f'{scene_id} is not present in {ann_file}')
    point_path = Path(info['lidar_points']['lidar_path'])
    if point_path.parent == Path('.'):
        point_path = Path('points') / point_path
    raw = np.fromfile(data_root / point_path, dtype=np.float32)
    if raw.size == 0 or raw.size % 6 != 0:
        raise ValueError(f'Invalid ScanNet [N, 6] point cloud: {data_root / point_path}')
    raw = raw.reshape(-1, 6)
    xyz_h = np.concatenate((raw[:, :3], np.ones((len(raw), 1), dtype=np.float32)), axis=1)
    align = np.asarray(info.get('axis_align_matrix', np.eye(4)), dtype=np.float32)
    points = (xyz_h @ align.T)[:, :3]
    colors = np.clip(raw[:, 3:6], 0, 255).astype(np.uint8)
    instances = info.get('instances', [])
    gt_boxes = np.asarray([item['bbox_3d'] for item in instances], dtype=np.float32)
    if gt_boxes.size == 0:
        gt_boxes = np.empty((0, 6), dtype=np.float32)
    gt_labels = np.asarray([item['bbox_label_3d'] for item in instances], dtype=np.int64)
    return points, colors, gt_boxes, gt_labels


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray,
              pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> None:
    vertices, vertex_colors, edges = [points], [colors], []
    vertex_offset = len(points)
    for boxes, color in ((pred_boxes, (255, 45, 45)), (gt_boxes, (45, 220, 70))):
        corners = box_corners(boxes)
        if len(corners) == 0:
            continue
        vertices.append(corners.reshape(-1, 3))
        vertex_colors.append(np.tile(np.array(color, dtype=np.uint8), (len(corners) * 8, 1)))
        for box_index in range(len(corners)):
            base = vertex_offset + box_index * 8
            edges.extend((base + start, base + end) for start, end in EDGES)
        vertex_offset += len(corners) * 8
    xyz, rgb = np.concatenate(vertices), np.concatenate(vertex_colors)
    with path.open('w', encoding='ascii') as handle:
        handle.write('ply\nformat ascii 1.0\n')
        handle.write(f'element vertex {len(xyz)}\n')
        handle.write('property float x\nproperty float y\nproperty float z\n')
        handle.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        handle.write(f'element edge {len(edges)}\nproperty int vertex1\nproperty int vertex2\nend_header\n')
        for point, color in zip(xyz, rgb):
            handle.write(f'{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {color[0]} {color[1]} {color[2]}\n')
        for start, end in edges:
            handle.write(f'{start} {end}\n')


def draw_boxes(axis, boxes: np.ndarray, color: str, labels: np.ndarray,
               scores: np.ndarray | None) -> None:
    for index, corners in enumerate(box_corners(boxes)):
        for start, end in EDGES:
            axis.plot(*corners[[start, end]].T, color=color, linewidth=1.3)
        label = int(labels[index]) if index < len(labels) else -1
        name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label)
        text = name if scores is None else f'{name} {scores[index]:.2f}'
        axis.text(*corners[6], text, color=color, fontsize=7)


def visualize_result(result_path: Path, ann_file: Path, args: argparse.Namespace,
                     output_dir: Path) -> dict:
    result = torch.load(result_path, map_location='cpu', weights_only=False)
    points, colors, gt_boxes, gt_labels = load_scene(
        result['scene_id'], args.data_root, ann_file)
    pred_mask = result['pred_scores'] >= args.score_thr
    pred_boxes = result['pred_boxes'][pred_mask]
    pred_labels = result['pred_labels'][pred_mask]
    pred_scores = result['pred_scores'][pred_mask]
    if len(points) > args.max_points:
        indices = np.linspace(0, len(points) - 1, args.max_points, dtype=np.int64)
        points, colors = points[indices], colors[indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(13, 10))
    axes = [figure.add_subplot(2, 2, index + 1, projection='3d') for index in range(3)]
    for axis, (elevation, azimuth, title) in zip(
            axes, ((25, -60, 'Perspective'), (90, -90, 'Top-down'), (0, -90, 'Front'))):
        axis.scatter(points[:, 0], points[:, 1], points[:, 2],
                     c=colors / 255., s=.35, alpha=.55)
        draw_boxes(axis, gt_boxes, 'lime', gt_labels, None)
        draw_boxes(axis, pred_boxes, 'red', pred_labels, pred_scores)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(title)
        axis.set_xlabel('x (m)'); axis.set_ylabel('y (m)'); axis.set_zlabel('z (m)')
    figure.tight_layout()
    figure.savefig(output_dir / 'scene_comparison.png', dpi=180)
    plt.close(figure)
    write_ply(output_dir / 'scene_comparison.ply', points, colors, pred_boxes, gt_boxes)
    summary = dict(scene_id=result['scene_id'], point_count=int(len(points)),
                   gt_count=int(len(gt_boxes)), pred_count=int(len(pred_boxes)),
                   score_threshold=args.score_thr)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True,
                        help='A result .pt file or a directory containing .pt files.')
    parser.add_argument('--data-root', type=Path,
                        default=Path('/mnt/workspace/data/ScanNet_processed'))
    parser.add_argument('--ann-file', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--max-points', type=int, default=60000)
    args = parser.parse_args()

    if args.result.is_file():
        result_paths = [args.result]
        output_dirs = [args.output_dir or args.result.with_suffix('')]
    elif args.result.is_dir():
        result_paths = sorted(args.result.glob('*.pt'))
        if not result_paths:
            raise FileNotFoundError(f'No .pt result files found in {args.result}')
        base_dir = args.output_dir or args.result / 'visualizations'
        output_dirs = [base_dir / path.stem for path in result_paths]
    else:
        raise FileNotFoundError(args.result)

    ann_file = args.ann_file or args.data_root / 'scannet_infos_val_pts.pkl'
    for result_path, output_dir in zip(result_paths, output_dirs):
        summary = visualize_result(result_path, ann_file, args, output_dir)
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
