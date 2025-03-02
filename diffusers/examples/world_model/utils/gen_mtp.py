import numpy as np
import cv2
import torch
import os
import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import pickle
from scipy import sparse

from pyquaternion import Quaternion
from PIL import Image

from einops import rearrange, repeat

# colors_map = torch.tensor(
#     [
#         [0, 0, 0, 255],  # unknown
#         [255, 158, 0, 255],  #  1 car  orange
#         [255, 99, 71, 255],  #  2 truck  Tomato
#         [255, 140, 0, 255],  #  3 trailer  Darkorange
#         [255, 69, 0, 255],  #  4 bus  Orangered
#         [233, 150, 70, 255],  #  5 construction_vehicle  Darksalmon
#         [220, 20, 60, 255],  #  6 bicycle  Crimson
#         [255, 61, 99, 255],  #  7 motorcycle  Red
#         [0, 0, 230, 255],  #  8 pedestrian  Blue
#         [47, 79, 79, 255],  #  9 traffic_cone  Darkslategrey
#         [112, 128, 144, 255],  #  10 barrier  Slategrey
#         [0, 207, 191, 255],  # 11  driveable_surface  nuTonomy green
#         [175, 0, 75, 255],  #  12 other_flat
#         [75, 0, 75, 255],  #  13  sidewalk
#         [112, 180, 60, 255],  # 14 terrain
#         [222, 184, 135, 255],  # 15 manmade Burlywood
#         [0, 175, 0, 255],  # 16 vegetation  Green

#         [0, 0, 0, 255],  # unknown

#     ]
# ).type(torch.uint8)

colors_map = torch.tensor(
    [
        [0, 0, 0, 255],
        [255, 120, 50, 255],  # barrier              orangey
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        [139, 137, 137, 255], # other_flat           dark grey
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green

        [0, 0, 0, 255],  # unknown

        # [0, 255, 127, 255],  # ego car              dark cyan
        # [255, 99, 71, 255],
        # [0, 191, 255, 255],
        # [125, 125, 125, 255]
    ]
).type(torch.uint8)





def concat_images(images, direction='horizontal', pad=0, pad_value=0):
    if len(images) == 1:
        return images[0]
    is_pil = isinstance(images[0], Image.Image)
    if is_pil:
        images = [np.array(image) for image in images]
    if direction == 'horizontal':
        height = max([image.shape[0] for image in images])
        width = sum([image.shape[1] for image in images]) + pad * (len(images) - 1)
        new_image = np.full((height, width, images[0].shape[2]), pad_value, dtype=images[0].dtype)
        begin = 0
        for image in images:
            end = begin + image.shape[1]
            new_image[: image.shape[0], begin:end] = image
            begin = end + pad
    elif direction == 'vertical':
        height = sum([image.shape[0] for image in images]) + pad * (len(images) - 1)
        width = max([image.shape[1] for image in images])
        new_image = np.full((height, width, images[0].shape[2]), pad_value, dtype=images[0].dtype)
        begin = 0
        for image in images:
            end = begin + image.shape[0]
            new_image[begin:end, : image.shape[1]] = image
            begin = end + pad
    else:
        assert False
    if is_pil:
        new_image = Image.fromarray(new_image)
    return new_image


def get_multiview_imgs(scene_name, idx,return_len=None, offset=0, nusc_infos=None, scale_intrinsic=None, idx_permute_img=None, image_size=None, view_order=None):
    pixel_values_list = []
    images_list = []
    cam_front_list, cam_front_right_list, cam_front_left_list, cam_back_list, cam_back_left_list, cam_back_right_list = [], [], [], [], [], []
    all_camera2ego_list, all_camera_intrinsics_list = [], []
    # front, front_right, front_left, back, back_left, back_right
    for i in range(return_len + offset):
        cur_nusc_info = nusc_infos[scene_name][idx+i]
        cur_camera2ego_list, cur_camera_intrinsics_list = [], []
        for _, camera_info in cur_nusc_info["cams"].items():
            # camera intrinsics
            # camera_intrinsics = np.eye(4).astype(np.float32)
            # camera_intrinsics[:3, :3] = camera_info["cam_intrinsic"]
            # cur_camera_intrinsics_list.append(camera_intrinsics)

            cur_camera_intrinsics_list.append(camera_info["cam_intrinsic"].astype(np.float32))

            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                camera_info["sensor2ego_rotation"]
            ).rotation_matrix
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
            cur_camera2ego_list.append(camera2ego)

        cur_camera_intrinsics = np.stack(cur_camera_intrinsics_list)
        cur_camera2ego = np.stack(cur_camera2ego_list)
        all_camera_intrinsics_list.append(cur_camera_intrinsics)
        all_camera2ego_list.append(cur_camera2ego)

    all_camera_intrinsics = np.stack(all_camera_intrinsics_list)
    all_camera2ego = np.stack(all_camera2ego_list)

    all_camera_intrinsics[:, :, 0, 0] = scale_intrinsic[0] * all_camera_intrinsics[:, :, 0, 0]
    all_camera_intrinsics[:, :, 0, 2] = scale_intrinsic[0] * all_camera_intrinsics[:, :, 0, 2]
    all_camera_intrinsics[:, :, 1, 1] = scale_intrinsic[1] * all_camera_intrinsics[:, :, 1, 1]
    all_camera_intrinsics[:, :, 1, 2] = scale_intrinsic[1] * all_camera_intrinsics[:, :, 1, 2]

    all_camera_intrinsics = np.transpose(all_camera_intrinsics, axes=(1, 0, 2, 3))
    all_camera2ego = np.transpose(all_camera2ego, axes=(1, 0, 2, 3))

    all_camera_intrinsics = all_camera_intrinsics[idx_permute_img]
    all_camera2ego = all_camera2ego[idx_permute_img]

    for i in range(return_len + offset):
        cur_nusc_info = nusc_infos[scene_name][idx+i]
        cam_front_list.append('.' + cur_nusc_info['cams']['CAM_FRONT']['data_path'])
        cam_front_right_list.append('.' + cur_nusc_info['cams']['CAM_FRONT_RIGHT']['data_path'])
        cam_front_left_list.append('.' + cur_nusc_info['cams']['CAM_FRONT_LEFT']['data_path'])
        cam_back_list.append('.' + cur_nusc_info['cams']['CAM_BACK']['data_path'])
        cam_back_left_list.append('.' + cur_nusc_info['cams']['CAM_BACK_LEFT']['data_path'])
        cam_back_right_list.append('.' + cur_nusc_info['cams']['CAM_BACK_RIGHT']['data_path'])

    all_cam_dict = {"CAM_FRONT_LEFT": cam_front_left_list, "CAM_FRONT": cam_front_list, "CAM_FRONT_RIGHT": cam_front_right_list, 
                    "CAM_BACK_RIGHT": cam_back_right_list, "CAM_BACK": cam_back_list, "CAM_BACK_LEFT": cam_back_left_list}
    for cam_name in view_order:
        video = all_cam_dict[cam_name]
        # frames = self.load_and_transform_frames(video, self.loader, self.img_transform, img_size=self.image_size)

        # frames = torch.cat(frames, 1) # c,t,h,w
        # frames = frames.transpose(0, 1) # t,c,h,w

        # frames_low = self.load_and_transform_frames(video, self.loader, self.img_transform, img_size=(288,160))
        # frames_low = torch.cat(frames_low, 1) # c,t,h,w
        # frames_low = frames_low.transpose(0, 1) # t,c,h,w
        
        # pixel_values_list.append(frames)

        img = Image.open(video[0])
        img = img.convert('RGB')
        img = img.resize(image_size)

        images_list.append(img)

    # pixel_values = torch.stack(pixel_values_list)
    images = images_list

    return images, all_camera_intrinsics, all_camera2ego

def get_sampling_points(origin_camera_size, down_sample, grid_config):
    ih, iw = origin_camera_size
    fh, fw = ih // down_sample, iw // down_sample
    d_bound = grid_config['d_bound']
    fd = int((d_bound[1] - d_bound[0]) // d_bound[2])
    x_coords = torch.linspace(0, iw - 1, fw)
    y_coords = torch.linspace(0, ih - 1, fh)
    d_coords = torch.linspace(d_bound[0], d_bound[1], fd)
    grid_d, grid_y, grid_x = torch.meshgrid(d_coords, y_coords, x_coords)
    sampled_points = torch.stack((grid_x, grid_y, grid_d), dim=-1)

    sampled_points = torch.cat([sampled_points[..., :2] * sampled_points[..., 2:3], 
                                sampled_points[..., 2:3], torch.ones_like(sampled_points[..., 2:3])], dim=-1) #(ws, hs, ds, 4)
    return sampled_points
    
def back_projection(volume_points, K, T):
    B, D, H, W, _ = volume_points.shape # BDHW4
    # cam2imgs = repeat(torch.eye(4).to(K), 'p q -> bs p q', bs=B)
    cam2imgs = repeat(torch.eye(4).to(K), 'p q -> bs p q', bs=B).clone()
    cam2imgs[:, :3, :3] = K
    # cam_points = rearrange(torch.inverse(cam2imgs), 'bs p q -> bs 1 1 1 p q') @ (volume_points/10).unsqueeze(-1)
    
    cam_points = rearrange(torch.inverse(cam2imgs), 'bs p q -> bs 1 1 1 p q') @ (volume_points).unsqueeze(-1)
        # cam_points = cam_points*10
    ego_points = rearrange(T, 'bs p q -> bs 1 1 1 p q') @ cam_points
    return ego_points


def generate_proj_sem_map(index, idx, return_len):
    # output = torch.randint(0, 20, (2, 10, 3))
    # output[:, 0:3] = 17
    # output[0, 0:10] = 17
    # mask = output != 17
    # mask_int = mask.to(torch.int8)
    # a0 = torch.argmax(mask_int, dim=1)


    data_path = '../data/nuscenes/'
    imageset = '../data/nuscenes_infos_train_temporal_v3_scene.pkl'
    input_dataset = 'gts'
    img_size = (576, 320)
    # return_len = 1

    with open(imageset, 'rb') as f:
        data = pickle.load(f)

    nusc_infos = data['infos']
    scene_names = list(nusc_infos.keys())
    data_path = data_path
    scale_intrinsic = (img_size[0] / 1600, img_size[1] / 900)

    idx_permute_img = np.array([2, 0, 1, 5, 3, 4])
    view_order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]


    # index = 0 # scene index
    # idx = 0 # start time step 
    # i = 0 # time step


    # index = index % len(nusc_infos)
    scene_name = scene_names[index]

    # 576, 320
    images, all_camera_intrinsics, all_camera2ego = get_multiview_imgs(scene_name, idx,return_len=return_len, offset=0, nusc_infos=nusc_infos, scale_intrinsic=scale_intrinsic, idx_permute_img=idx_permute_img, image_size=img_size, view_order=view_order)

    all_img_fov_list, all_img_list = [], []

    for i in range(return_len):


        cur_camera_intrinsics = all_camera_intrinsics[:, i:i+1]
        cur_camera2ego = all_camera2ego[:, i:i+1]


        token = nusc_infos[scene_name][idx + i]['token']
        label_file = os.path.join(data_path, f'{input_dataset}/{scene_name}/{token}/labels.npz')
        label = np.load(label_file)
        occ = label['semantics']


        # Config
        origin_occ_shape = (16, 200, 200)
        origin_camera_size = (320, 576)
        # down_sample = 8
        down_sample = 1

        grid_config = dict(
            x_bound = [-40.0, 40.0, 0.4],
            y_bound = [-40.0, 40.0, 0.4],
            z_bound = [-1.0, 5.4, 0.4],
            d_bound = [0.5, 48.5, 1.0],
        )
        ###

        occ = np.repeat(occ[None, ...], 6, axis=0)
        occ = np.transpose(occ, axes=(0, 3, 1, 2))
        occ = occ[:, None, ...]

        occ = torch.tensor(occ)
        # K = torch.tensor(all_camera_intrinsics).permute(1, 0, 2, 3)
        # T = torch.tensor(all_camera2ego).permute(1, 0, 2, 3)

        K = torch.tensor(cur_camera_intrinsics).permute(1, 0, 2, 3)
        T = torch.tensor(cur_camera2ego).permute(1, 0, 2, 3)

        T = T.reshape(-1, T.shape[2], T.shape[3])  # B,3,4,4
        K = K.reshape(-1, K.shape[2], K.shape[3])  # B*3,3,3

        B, C, D_occ, H_occ, W_occ,  = occ.shape
        D_oi, H_oi, W_oi = origin_occ_shape
        ratio = (W_oi / W_occ, H_oi / H_occ, D_oi / D_occ)

        ego2egofeat = torch.eye(4)
        ego2egofeat[0,0] = 1 / grid_config['x_bound'][2] / ratio[0]
        ego2egofeat[1,1] = 1 / grid_config['y_bound'][2] / ratio[1]
        ego2egofeat[2,2] = 1 / grid_config['z_bound'][2] / ratio[2]
        ego2egofeat[0,3] = - grid_config['x_bound'][0] / grid_config['x_bound'][2] / ratio[0]
        ego2egofeat[1,3] = - grid_config['y_bound'][0] / grid_config['y_bound'][2] / ratio[1]
        ego2egofeat[2,3] = - grid_config['z_bound'][0] / grid_config['z_bound'][2] / ratio[2]

        ego2egofeat = repeat(ego2egofeat, 'p q -> bs p q', bs=B).to(K)

        sampled_points = get_sampling_points(origin_camera_size, down_sample, grid_config) #(ds, hs, ws, 4)
        sampled_points = repeat(sampled_points, 'ds hs ws d -> b ds hs ws d', b=B)
        ego_points = back_projection(sampled_points, K, T)
        egofeat_points = rearrange(ego2egofeat, 'bs p q -> bs 1 1 1 p q') @ ego_points
        egofeat_points = egofeat_points.squeeze(-1)[...,:3] #(6, 48, 40, 72, 3)

        origin_occ = torch.tensor(label['semantics'])
        
        output = torch.full(egofeat_points.shape[:-1], 17, dtype=torch.uint8)  
        x = egofeat_points[..., 0]  
        y = egofeat_points[..., 1]  
        z = egofeat_points[..., 2]  

        valid_mask = (x >= 0) & (x < 200) & (y >= 0) & (y < 200) & (z >= 0) & (z < 16)  

        x_floor = torch.floor(x).long()  
        y_floor = torch.floor(y).long()  
        z_floor = torch.floor(z).long()

        output[valid_mask] = origin_occ[x_floor[valid_mask], y_floor[valid_mask], z_floor[valid_mask]]  


        mask = output != 17
        mask_int = mask.to(torch.int8)
        select_near_index = torch.argmax(mask_int, dim=1)


        B, height, width = select_near_index.shape

        select_output = output.permute(0, 2, 3, 1).reshape(-1, 48)[np.arange(B * height * width), select_near_index.reshape(-1)]
        # select_output = select_output.reshape(B, height, width)

        col_pcl = torch.index_select(colors_map, 0, select_output.type(torch.int32))
        img_fov = col_pcl[:, :3].reshape(B, height, width, 3)


        # image_sem = Image.fromarray(img_fov[0].numpy())  
        # image_sem.save('output_image_sem.png')  

        # images[0].save('resize_image_rgb.png')

        all_img_fov_list.append(img_fov.numpy())
        all_img_list.append(images)



    for idx_frame in range(return_len):
        cur_multiview_imgs = []
        cur_origin_imgs = []
        for idx_cam in idx_permute_img:
            cur_multiview_imgs.append(all_img_fov_list[idx_frame][idx_cam])
            cur_origin_imgs.append(all_img_list[idx_frame][idx_cam])

        cur_multiview_imgs = all_img_fov_list[idx_frame]
        cur_origin_imgs = all_img_list[idx_frame]

        cur_row_first = concat_images(cur_multiview_imgs[:3], pad=0)
        cur_row_last = concat_images(cur_multiview_imgs[3:], pad=0)
        cat_cur_multiview_imgs = concat_images([cur_row_first, cur_row_last], direction='vertical')
        Image.fromarray(cat_cur_multiview_imgs).save('sem_{:06d}_{}.jpg'.format(idx, idx_frame))

        cur_row_first = concat_images(cur_origin_imgs[:3], pad=0)
        cur_row_last = concat_images(cur_origin_imgs[3:], pad=0)
        cat_cur_multiview_imgs = concat_images([cur_row_first, cur_row_last], direction='vertical')
        cat_cur_multiview_imgs.save('origin_{:06d}_{}.jpg'.format(idx, idx_frame))


if __name__ == "__main__":
    generate_proj_sem_map(0, 0, 2)