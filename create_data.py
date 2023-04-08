# import os, sys
import torch, json
import numpy as np

from main import build_model_main
from dino.util.slconfig import SLConfig
from dino.datasets import build_dataset
from dino.util.visualizer import COCOVisualizer
from dino.util import box_ops
from torchvision.ops import box_iou
from PIL import Image
import torchvision.transforms.functional as F


model_config_path = "configs/DINO/DINO_4scale_swin.py" # change the path of the model config file
model_checkpoint_path = "./logs/with_pose/checkpoint.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.

args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
args.backbone_dir = './checkpoints/'
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

args.dataset_file = 'coco'
args.coco_path = "./data/CityUHK-X-BEV/det" # the path of coco
args.fix_size = False
dataset_train = build_dataset(image_set='train_reg', args=args)  
dataset_eval = build_dataset(image_set='val', args=args)  

idx=0
pos_idx = 0
neg_idx = 0
width_threshold = 0.022
height_threshold = 0.065
data = {}
im_w = 1066
im_h = 800


for _, targets in dataset_train:
    fp = f"data/CityUHK-X-BEV/det/train_{idx}.png"
    img = Image.open(fp).resize((im_w,im_h))

    gt_boxes = targets["boxes"]

    valid_idx = (gt_boxes[:,2]>0.5*width_threshold)&(gt_boxes[:,2]>0.5*height_threshold)&\
                (gt_boxes[:,2]<2*width_threshold)&(gt_boxes[:,2]<2*height_threshold)
    gt_boxes = gt_boxes[valid_idx]
    # gt_boxes[:,2:] *= 2
    gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes).to("cuda")

    gt_boxes[:,0] *=im_w
    gt_boxes[:,1] *=im_h
    gt_boxes[:,2] *=im_w
    gt_boxes[:,3] *=im_h

    for i in range(min(5,len(gt_boxes))):
        # positive
        bounding = gt_boxes[i].cpu().numpy().tolist()
        area = img.crop(bounding).resize((32,72))
        area.save(f"data_postprocess/pos_data/{pos_idx}.png")
        pos_idx += 1
        # negative
        x = torch.randint(0, im_w-100, size=(1,)).item()
        y = torch.randint(0, im_h-100, size=(1,)).item()
        # height, width
        bh = bounding[3] - bounding[1]
        bw = bounding[2] - bounding[0]
        box = torch.Tensor([x,y, x+bw, y+bh]).cuda()
        if torch.all(box_iou(box[None,:], gt_boxes)<0.2):
            area = img.crop([x,y, x+bw, y+bh]).resize((32,72))
            area.save(f"data_postprocess/neg_data/{neg_idx}.png")
            neg_idx += 1
    idx+=1

idx = 0
for image, _ in dataset_eval:
    output = model.cuda()(image[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0] 
    thershold = 0.5 # set a thershold

    scores = output['scores']
    labels = output['labels']
    select_mask = scores > thershold
    pred_boxes = output['boxes'][select_mask]
    pred_scores = output['scores'][select_mask]
    save_ = {}
    save_["pred"] = pred_boxes.cpu().numpy().tolist()

    min_thershold = 0.1
    remain_boxes = output['boxes'][(scores < thershold)&(scores > min_thershold)]
    isMiss_list = torch.zeros(len(remain_boxes), dtype=bool)
    for i, box in enumerate(remain_boxes):
        isMiss = torch.all(box_iou(box[None,:], pred_boxes)<0.1)
        isMiss_list[i] = isMiss
        if isMiss:
            pred_boxes = torch.cat([pred_boxes,box[None,:]])
    save_["remain"] = remain_boxes[isMiss_list].cpu().numpy().tolist()

    data[f"{idx}"] = save_
    idx += 1
    print(idx)
    
with open('data_postprocess/valid_boxes.json', 'w') as f:
        json.dump(data, f, indent=2)
