# import os, sys
import torch, json
import numpy as np

from dino.util.slconfig import SLConfig
from dino.datasets import build_dataset
from dino.util.visualizer import COCOVisualizer
from dino.util import box_ops
from torchvision.ops import box_iou
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
torch.manual_seed(42)

def build_dataval():
    model_config_path = "configs/DINO/DINO_4scale_swin.py" # change the path of the model config file
    model_checkpoint_path = "./logs/with_pose/checkpoint.pth" # change the path of the model checkpoint

    args = SLConfig.fromfile(model_config_path) 
    args.device = 'cuda' 
    args.backbone_dir = './checkpoints/'

    args.dataset_file = 'coco'
    args.coco_path = "./data/CityUHK-X-BEV/det" # the path of coco
    args.fix_size = False

    dataset_val = build_dataset(image_set='val', args=args)   
    return dataset_val

def classify(arr, image):
    # arr[:,2:] *= 2
    arr = box_ops.box_cxcywh_to_xyxy(arr)
    im_w = 1066
    im_h = 800
    arr[:,0] *= im_w
    arr[:,1] *= im_h
    arr[:,2] *= im_w
    arr[:,3] *= im_h
    valid_idx = []
    arr = arr.cpu().numpy()
    j=0
    for bounding in arr:
        img = image.crop(bounding).resize((32,72))
        input = preprocess(img).to("cuda").unsqueeze(0)
        out = model(input).argmax()
        valid_idx.append(out)
        j+=1
    valid_idx = torch.Tensor(valid_idx)==1
    return valid_idx

def plot_boxes(image, boxes, rem, name):
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)
    for box in rem:
        box = box.cpu().numpy().tolist()
        box[0] *= im_w
        box[1] *= im_h
        box[2] *= im_w
        box[3] *= im_h
        # Create a Rectangle patch
        rect = patches.Rectangle((box[0]-box[2]/2,box[1]-box[3]/2), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
    
    for box in boxes:
        box = box.cpu().numpy().tolist()
        box[0] *= im_w
        box[1] *= im_h
        box[2] *= im_w
        box[3] *= im_h
        # Create a Rectangle patch
        rect = patches.Rectangle((box[0]-box[2]/2,box[1]-box[3]/2), box[2], box[3], linewidth=1, edgecolor='blue', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
    '''
    '''
    plt.savefig("data_postprocess/visual/"+name)
    plt.close()

f = open('data_postprocess/valid_boxes.json','r')
data = json.load(f)
model=torch.load("data_postprocess/ckpts/epoch_4.pth").to("cuda")
model.eval()
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


coco_path = "data/CityUHK-X-BEV/det/"
im_w = 1066
im_h = 800
sz = torch.Tensor((800, 1066))

for i in range(507):
    image = Image.open(coco_path+f"test_{i}.png").convert('RGB').resize((1066,800))
    boxes = torch.Tensor(data[f"{i}"]["pred"]) 
    rem = torch.Tensor(data[f"{i}"]["remain"])
    boxes = box_ops.box_xyxy_to_cxcywh(boxes)
    if len(rem)==0:
        continue
    rem = box_ops.box_xyxy_to_cxcywh(rem)

    valid_idx = classify(rem.clone(), image)
    rem = rem[valid_idx]

    print(i)
    plot_boxes(image, boxes, rem,  f"{i}.png")