cd /mnt/workspace/code/dudet/VGGT-Det-CVPR2026

python tools/visualize_scannet_gt.py \
  --split train \
  --scene-id scene0005_00 \
  --output-dir /mnt/workspace/output/scannet_vis


cd /mnt/workspace/code/dudet/vggt

python reconstruct_scannet_scene.py \
  --scene-id scene0013_01 \
  --output-dir /mnt/workspace/output/vggt \
  --device cuda \
  --max-images 81

bash tools/dist_test.sh projects/DETR3D/configs/detr3d_scannet_rgb_only.py work_dirs/detr3d_scannet_rgb_only/epoch_4.pth 1


python tools/visualize_scannet_detr3d_result.py \
  --result work_dirs/detr3d_scannet_rgb_only/scene_results/scene0568_00.pt \
  --data-root /mnt/workspace/data/ScanNet_processed \
  --output-dir /mnt/workspace/output/3ddetr \
  --score-thr 0.05