# DINO

ssh zzhang452@144.214.121.21

python main.py   --output_dir logs/DINO/eval2     -c configs/DINO/DINO_4scale_swin.py --coco_path ./  --eval      --resume logs/DINO/R50-MS4/checkpoint0019.pth     --options dn_scalar=100 embed_init_tgt=TRUE     dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False     dn_box_noise_scale=1.0 backbone_dir=./checkpoints/