# DINO

ssh zzhang452@144.214.121.21

# train
# 25 epochs
python main.py --output_dir logs/train1_demo -c configs/DINO/DINO_4scale_swin.py --coco_path ./ --pretrain_model_path ./checkpoints/checkpoint0029_4scale_swin.pth --options dn_scalar=100 embed_init_tgt=TRUE  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False  dn_box_noise_scale=1.0 backbone_dir=./checkpoints --use_dino_pertrained

# 5 epochs
python main.py --output_dir logs/with_pose_demo -c configs/DINO/DINO_4scale_swin.py --coco_path ./ --pretrain_model_path logs/train1/checkpoint.pth --options dn_scalar=100 embed_init_tgt=TRUE  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False  dn_box_noise_scale=1.0 backbone_dir=./checkpoints

# eval
python main.py   --output_dir logs/DINO/eval1 -c configs/DINO/DINO_4scale_swin.py --coco_path ./  --eval      --resume logs/train1/checkpoint.pth     --options dn_scalar=100 embed_init_tgt=TRUE     dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False     dn_box_noise_scale=1.0 backbone_dir=./checkpoints/

# test
modify DINO_4scale_swin.py/ FUNC: trained_model
python test.py  --task-option-file configs/option.yaml --use-gpus 0

python run_metrics.py         --task-option-file configs/option.yaml         --model-output-file logs/test/DINO/Apr08_20-55-17/test/model-output.h5         --output-csv logs/test/DINO/Apr08_20-55-17/test/metric_result.csv         --use-gpu 0

# post processing
python create_data.py
python train_classifier.py
python verify.py

# description
Feb24_20-20-59: without additional postprocess
Mar09_01-13-47: adding log confidence prediction without classifier
Mar09_00-57-53: adding log confidence prediction with classifier
Mar19_14-50-23: without suppression & with classifier
Mar19_15-05-50: without suppression