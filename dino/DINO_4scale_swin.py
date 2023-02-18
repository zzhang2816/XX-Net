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
    args.model_checkpoint_path = "./logs/DINO/train1/checkpoint0029.pth" 
    args.device = 'cuda' 
    args.backbone_dir = './checkpoints/'
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(args.model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    return model,postprocessors
    # output = model.cuda()(image[None].cuda())
    # output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]