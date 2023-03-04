# DINO

ssh zzhang452@144.214.121.21

# train
# 25 epochs
python main.py --output_dir logs/train1 -c configs/DINO/DINO_4scale_swin.py --coco_path ./ --pretrain_model_path ./checkpoints/checkpoint0029_4scale_swin.pth --options dn_scalar=100 embed_init_tgt=TRUE  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False  dn_box_noise_scale=1.0 backbone_dir=./checkpoints

# 5 epochs
python main.py --output_dir logs/with_pose -c configs/DINO/DINO_4scale_swin.py --coco_path ./ --pretrain_model_path logs/train1/checkpoint.pth --options dn_scalar=100 embed_init_tgt=TRUE  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False  dn_box_noise_scale=1.0 backbone_dir=./checkpoints

# eval
python main.py   --output_dir logs/DINO/eval1 -c configs/DINO/DINO_4scale_swin.py --coco_path ./  --eval      --resume ./logs/with_pose/checkpoint.pth     --options dn_scalar=100 embed_init_tgt=TRUE     dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False     dn_box_noise_scale=1.0 backbone_dir=./checkpoints/

# test
modify DINO_4scale_swin.py/ FUNC: trained_model
python test.py  --task-option-file configs/option.yaml --use-gpus 0

python run_metrics.py         --task-option-file configs/option.yaml         --model-output-file logs/test/DINO/Feb24_21-48-04/test/model-output.h5         --output-csv logs/test/DINO/Feb24_21-48-04/test/metric_result.csv         --use-gpu 0
