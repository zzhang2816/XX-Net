import os, sys
import torch, json
import numpy as np

# from main import build_model_main
from dino.util.slconfig import SLConfig
# from datasets import build_dataset

def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from dino.models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors
    
def trained_model(model_config_path):
    args = SLConfig.fromfile(model_config_path) 
    args.model_checkpoint_path = "./logs/with_pose/checkpoint.pth" 
    args.device = 'cuda' 
    args.backbone_dir = './checkpoints/'
    model, criterion, postprocessors = build_model_main(args)
    use_post_processing = args.use_post_processing
    checkpoint = torch.load(args.model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    return model,postprocessors, use_post_processing
    # output = model.cuda()(image[None].cuda())
    # output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

def trained_classifier():
    torch.manual_seed(42)
    model=torch.load("data_postprocess/ckpts/epoch_4.pth").to("cuda")
    model.eval()
    return model