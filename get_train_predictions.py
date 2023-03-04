import torch, json
import numpy as np

from main import build_model_main
from dino.util.slconfig import SLConfig
from dino.datasets import build_dataset
from dino.util.visualizer import COCOVisualizer
from dino.util import box_ops
from dino.models.dino.matcher import HungarianMatcher
from torchvision.ops import box_iou


model_config_path = "configs/DINO/DINO_4scale_swin.py" # change the path of the model config file
model_checkpoint_path = "./logs/with_pose/checkpoint.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.


args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
args.backbone_dir = './checkpoints/'
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')

model.load_state_dict(checkpoint['model'])

model.training = False

args.dataset_file = 'coco'
args.coco_path = "./" 
args.fix_size = False

dataset_train = build_dataset(image_set='val', args=args)   

analysis = []

for image, targets in dataset_train:
    with torch.no_grad():
        output = model.cuda()(image[None].cuda())
    print(output["height"])
    print(targets["height"])
    print("===================================")
    
    # targets = [{k: v.cuda() for k, v in t.items()} for t in [targets]]
    # focal_alpha = 0.25
    # set_cost_class = 0
    # set_cost_bbox = 5.0
    # set_cost_giou = 2.0
    # matcher = HungarianMatcher(
    #         cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou,
    #         focal_alpha=focal_alpha
    #     )
    # indices = matcher(outputs_without_aux, targets)
    # print(output["pred_boxes"])
    # postive = indices[0][0]
    # print(indices)
    # break

   