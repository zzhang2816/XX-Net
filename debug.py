import torch
import pickle

# from models import BEVTransform
    

anno_path = "data/CityUHK-X-BEV/det/camera_data/my_train_list.pkl"
train_d = pickle.load(open(anno_path, 'rb'))
# train_d = train_d[0]
print(train_d[0].keys())
res = torch.Tensor([train_d[i]["camera_angle"] for i in range(len(train_d))])
print(res.min(),res.max())
res = torch.Tensor([train_d[i]["camera_height"] for i in range(len(train_d))])
print(res.min(),res.max())

# tmp=train_d[1608]['feet'][0,:]
# print(len(tmp[tmp>0]))


# anno1_path = "data/CityUHK-X-BEV/det/train_list.pkl"
# train_d1 = pickle.load(open(anno1_path, 'rb'))
# annos = train_d1[1608]['annotations']
# bboxes = torch.Tensor([anno['bbox'] for anno in annos])
# print(len(bboxes))

# print(bboxes[:,0]/512)
# annos = [train_d1[i]['annotations'] for i in range(len(train_d1))]
# total = 0
# for i,anno in enumerate(annos):
#     bboxes = [anno[j]['bbox'] for j in range(len(anno))]
#     feet_x = [(box[0]+box[2])/2 for box in bboxes]
#     feet_y = [box[3] for box in bboxes]
#     feet_pixels = torch.zeros(1, 3, 121)
#     feet_pixels[:, 2] = 1
#     feet_pixels[0, 0, :len(bboxes)] = torch.Tensor(feet_x)
#     feet_pixels[0, 1, :len(bboxes)] = torch.Tensor(feet_y)
#     feet_pixels_gt = train_d[i]['feet']
#     total+=torch.sum(torch.abs(feet_pixels-feet_pixels_gt))

# print(total)


# length = len(train_d)
# feet_pixels = torch.stack([train_d[i]['feet'] for i in range(length)])
# world_coord = torch.stack([train_d[i]['world_coord'] for i in range(length)])


# im_size = (384, 512)
# bev_transform = BEVTransform()
# cam_h = torch.stack([train_d[i]['camera_height'] for i in range(length)])
# cam_a = torch.stack([train_d[i]['camera_angle'] for i in range(length)])
# camera_fu = torch.stack([train_d[i]['camera_fu'] for i in range(length)])
# camera_fv = torch.stack([train_d[i]['camera_fv'] for i in range(length)])



# i2w_mats, _, _ = bev_transform.get_bev_param(
#     im_size, cam_h, cam_a, camera_fu, camera_fv, w2i=False
# )
# pred_world_coords_homo = bev_transform.image_coord_to_world_coord(
#         feet_pixels, i2w_mats
# )

# print(pred_world_coords_homo-world_coord)