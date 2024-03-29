from collections import OrderedDict
from collections import defaultdict

import h5py
import numpy as np
import os
import torch
import torchvision.transforms.functional as F
from dino.util import box_ops
from torchvision import transforms
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.engine import DefaultTrainer
# from detectron2.engine import launch
# from detectron2.modeling import build_model
# from detectron2.structures import Boxes
# from detectron2.structures import Instances
# from detectron2.utils.visualizer import Visualizer

from models import BEVTransform
from models.metrics.kernels import GaussianKernel
from pytorch_helper.launcher.launcher import LauncherTask
from pytorch_helper.task import Batch
from pytorch_helper.utils.dist import get_rank
from pytorch_helper.utils.io import make_dirs
from pytorch_helper.utils.io import save_dict_as_csv
from pytorch_helper.utils.log import get_datetime
from pytorch_helper.utils.log import get_logger
from pytorch_helper.utils.log import pbar
from pytorch_helper.utils.meter import Meter
from pytorch_helper.settings.spaces import Spaces

from dino.DINO_4scale_swin import trained_model, trained_classifier
from torchvision.ops import box_iou
from PIL import Image
logger = get_logger(__name__)


@Spaces.register(Spaces.NAME.TASK, ['DINOTask', 'DINO2BEVTask'])
class DINOTask(LauncherTask):

    def __init__(self, task_option):
        super(DINOTask, self).__init__()
        self._option = task_option
        self.cuda_ids = task_option.cuda_ids
        self.rank = get_rank()
        self.model_config_path = task_option.dino_config_file

        self.trainer = None
        if not self._option.train:
            self.output_path_test_net_output = os.path.join(
                self._option.output_path_test, 'net_output'
            )
            make_dirs(self._option.output_path_test)
            make_dirs(self.output_path_test_net_output)

            torch.cuda.set_device(self.cuda_ids[0])

            # cfg = self.option.setup_cfg()
            # self.model = build_model(cfg)
            # checkpointer = DetectionCheckpointer(self.model)
            # checkpointer.load(cfg.MODEL.WEIGHTS)
            self.model, self.postprocessors, _ = trained_model(self.model_config_path)
            self.classifier = trained_classifier()
            self.loss_fn = self.option.loss.build()

            # self.use_inferred_pose = not self.option.test_option.pose_oracle
            # if self.use_inferred_pose:
            #     self.pose_net = self.option.pose_net.build()[0]
            #     self.pose_net.cuda()
            #     self.pose_net.eval()
            self.bev_transform = BEVTransform()

            sigma = 5
            ks = sigma * 6 + 1
            self.gaussian_conv = GaussianKernel(ks, sigma, channels=1)
            self.gaussian_conv.cuda()

            self.dataloader = self.option.dataloader.build()
            self.test_loader = self.dataloader.test_loader
            self.test_loader.dataset.do_normalization = False

            self.meter = Meter()
            self.in_stage_meter_keys = set()
            self.model_output_dict = defaultdict(list)

    @property
    def is_rank0(self):
        return self.rank == 0

    @property
    def is_distributed(self):
        return False

    @property
    def option(self):
        return self._option

    def train(self):
        pass
        # self.trainer = DefaultTrainer(self.option.setup_cfg())
        # self.trainer.resume_or_load(resume=self.option.resume)
        # self.trainer.train()

    def run(self):
        if self._option.train:
            pass
            # if len(self.cuda_ids) > 1:
            #     launch(
            #         main_func=self.train,
            #         num_gpus_per_machine=len(self.cuda_ids),
            #         dist_url=f'tcp://localhost:{int(os.environ["DDP_PORT"])}'
            #     )
            # else:
            #     self.train()
        else:
            self.model.train()
            self.model.training = False
            for batch in pbar(self.test_loader, desc='Test'):
                with torch.no_grad():
                    result = self.model_forward(batch)
                
                self.update_logging_in_stage(result)

            summary = self.summarize_logging_after_stage()

            path = os.path.join(self._option.output_path_test,
                                'test-summary.csv')
            save_dict_as_csv(path, summary)

            if self.option.test_option.save_model_output:
                path = os.path.join(self._option.output_path_test,
                                    f'model-output.h5')
                logger.info(f'Saving model output to {path}')
                h5file = h5py.File(path, 'w')
                for key, value in summary.items():
                    h5file.attrs[key] = value
                h5file.attrs['summary-ordered-keys'] = list(summary.keys())
                h5file.attrs['datetime_test'] = get_datetime()
                for name, data in self.model_output_dict.items():
                    logger.info(f'Saving {name}')
                    data = np.concatenate(data, axis=0)
                    h5file.create_dataset(
                        name, data.shape, data.dtype, data, compression='gzip'
                    )
                h5file.close()

    def model_forward(self, batch: dict):
        for k, v in batch.items():
            batch[k] = v.cuda()

        images = batch['image']
        gt_bev_map = batch['bev_map']
        bs, _, height, width = gt_bev_map.shape
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
     
        images = F.resize(images,size = [800, 1333])
        raw_image = images[0].clone()
        raw_image = (raw_image*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
        raw_image = Image.fromarray(raw_image)
        images = F.normalize(images, mean=mean, std=std)
        
        
        orig_target_sizes = torch.stack([torch.Tensor([height, width]) for _ in range(len(images))], dim=0)
        outputs = self.model.cuda()(images.cuda())

        outputs_height = outputs['height']
        outputs_angle = outputs['angle']
        outputs = self.postprocessors['bbox'](outputs, orig_target_sizes.cuda())
        thershold = 0.5 # set a thershold
        min_thershold = 0.1
        
        person_bboxes = []
        for output in outputs:
            scores = output['scores']
            pred_boxes = output['boxes'][scores>thershold]
            high_conf_len = len(pred_boxes)
            remain_boxes = output['boxes'][(scores < thershold)&(scores > min_thershold)]

            for box in remain_boxes:
                isMiss = torch.all(box_iou(box[None,:], pred_boxes)<0.2)
                if isMiss:
                    pred_boxes = torch.cat([pred_boxes,box[None,:]])
            valid_idx = classify(pred_boxes[high_conf_len:], raw_image, self.classifier)
            miss_preds = pred_boxes[high_conf_len:][valid_idx]
            pred_boxes = pred_boxes[:high_conf_len]
            if miss_preds.shape[0]>0:
                pred_boxes = torch.cat([pred_boxes, miss_preds])
            
            person_bboxes.append(pred_boxes)


        # person_bboxes = [
        #     output["boxes"][output['scores'] > thershold]
        #     for output in outputs
        # ]

        # self.visualize(images[0], person_bboxes[0])
        # pred.pred_boxes.tensor[pred.pred_classes == 0]
        pred = dict(person_bboxes=person_bboxes)
        camera_paras = {'camera_height': outputs_height*20, 'camera_angle': outputs_angle*1.2}

        pred.update(**camera_paras)

        result = Batch(gt=batch, pred=pred)
        n_coords = [len(bboxes) for bboxes in person_bboxes]
        bs, _, height, width = result.gt['bev_map'].size()

        # process outputs from detection model
        pred_feet_map = torch.zeros((bs, 1, height, width)).cuda()
        pred_head_map = torch.zeros((bs, 1, height, width)).cuda()
        feet_pixels = torch.zeros(bs, 3, max(n_coords)).cuda()
        feet_pixels[:, 2] = 1
        for i, bboxes in enumerate(person_bboxes):
            # feet position = center of bottom bbox border
            feet_pixels[i, 0, :len(bboxes)] = (bboxes[:, 0] + bboxes[:, 2]) / 2
            feet_pixels[i, 1, :len(bboxes)] = bboxes[:, 3]

            u = feet_pixels[i, 0, :len(bboxes)].long()
            v = feet_pixels[i, 1, :len(bboxes)].long()
            mask = (0 <= v) & (0 <= u) & (v < height) & (u < width)
            u = u[mask]
            v = v[mask]
            pred_feet_map[i, 0, v, u] = 1.

            u = ((bboxes[:, 0] + bboxes[:, 2]) / 2).long()
            v = bboxes[:, 1].long()
            mask = (0 <= v) & (0 <= u) & (v < height) & (u < width)
            u = u[mask]
            v = v[mask]
            pred_head_map[i, 0, v, u] = 1.
        pred_feet_map = self.gaussian_conv(pred_feet_map)
        pred_head_map = self.gaussian_conv(pred_head_map)
        result.pred['feet_map'] = pred_feet_map
        result.pred['head_map'] = pred_head_map

        # image view detection to world coordinates
        im_size = (height, width)
        # key = 'pred' if self.use_inferred_pose else 'gt'
        cam_h = result.pred['camera_height']
        cam_a = result.pred['camera_angle']
        camera_fu = result.gt['camera_fu']
        camera_fv = result.gt['camera_fv']
        i2w_mats, scales, centers = self.bev_transform.get_bev_param(
            im_size, cam_h, cam_a, camera_fu, camera_fv, w2i=False
        )
        result.pred['bev_scale'] = scales
        result.pred['bev_center'] = centers

        pred_world_coords_homo = self.bev_transform.image_coord_to_world_coord(
            feet_pixels, i2w_mats
        )
        pred_world_coords = [
            coord[:2, :n].cpu()
            for coord, n in zip(pred_world_coords_homo, n_coords)
        ]
        # evaluate the region of interest inside the BEV map only
        pred_bev_coords = self.bev_transform.world_coord_to_bev_coord(
            im_size, pred_world_coords_homo, scales, centers
        )
        pred_bev_map = torch.zeros((bs, 1, height, width)).cuda()

        # filter predicted coords
        pred_coords = []
        for i, (world_coords, bev_coords, n_annos) in enumerate(
            zip(pred_world_coords, pred_bev_coords, n_coords)
        ):
            world_coords = world_coords[:2, :int(n_annos)]
            bev_coords = bev_coords[:2, :int(n_annos)]
            roi = (bev_coords[0] >= 0) & (bev_coords[0] < width) & \
                  (bev_coords[1] >= 0) & (bev_coords[1] < height)
            roi = roi.cpu()
            pred_coords.append(world_coords[:, roi])
            for u, v in bev_coords[:, roi].int().T:
                pred_bev_map[i, 0, v.item(), u.item()] = 1
        pred_bev_map = self.gaussian_conv(pred_bev_map)
        result.pred['bev_map'] = pred_bev_map

        # loss
        result.loss = self.loss_fn(result.pred, result.gt)
        result.size = bs
        return result
        

    def summarize_logging_after_stage(self):
        summary = OrderedDict()
        summary['name'] = self.option.name
        summary['datetime'] = self.option.datetime
        summary['epoch'] = 'NA'
        summary['pth_file'] = 'NA'
        for key in sorted(list(self.in_stage_meter_keys)):
            summary[key] = self.meter.mean(key)
        return summary

    def update_logging_in_stage(self, result):
        loss = result.loss
        for k, v in loss.items():
            if v is not None:
                key = f'test/{k}-loss'
                self.meter.record(
                    tag=key, value=v.item(), weight=result.size,
                    record_op=Meter.RecordOp.APPEND,
                    reduce_op=Meter.ReduceOp.SUM
                )
                self.in_stage_meter_keys.add(key)

        # if self.option.test_option.save_im:
        #     self.save_visualization(result)
        if self.option.test_option.save_model_output:
            for key in ['scene_id', 'image_id']:
                self.model_output_dict.setdefault(key, list()).append(
                    result.gt[key].cpu().numpy()
                )
            for key, data in result.pred.items():
                if isinstance(data, torch.Tensor):
                    self.model_output_dict.setdefault(key, list()).append(
                        data.cpu().numpy()
                    )
                    
    def backup(self, immediate, resumable):
        if self.trainer:
            self.trainer.checkpointer.save(f'iter_{self.trainer.iter}')

    # def visualize(self, image, bboxes):
    #     print(image)
    #     from util import box_ops
    #     from util.visualizer import COCOVisualizer
    #     boxes = box_ops.box_xyxy_to_cxcywh(bboxes)
    #     pred_dict = {
    #         'image_id': 0,
    #         'boxes': boxes.cpu(),
    #         'size': torch.Tensor([1.0, 1.0]),
    #         'box_label': ['person' for _ in range(len(boxes))]
    #     }
    #     # print(image.get_device(),boxes.get_device(),target_sizes.get_device())
    #     vslzr = COCOVisualizer()
    #     vslzr.visualize(image.cpu(), pred_dict, savedir="./")

def classify(arr, image, model):
    valid_idx = []
    arr = arr.cpu().numpy()
    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for bounding in arr:
        input = image.crop(bounding).resize((32,72))
        input = preprocess(input).cuda().unsqueeze(0)
        out = model(input).argmax()
        valid_idx.append(out)
    valid_idx = torch.Tensor(valid_idx)==1
    return valid_idx
