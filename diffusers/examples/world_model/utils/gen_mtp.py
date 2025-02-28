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

# data_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
# # data_file = './data/nuscenes/nuscenes_occ_infos_train.pkl'

# with open(data_file, "rb") as file:
#     nus_pkl = pickle.load(file)



# dataroot = "./data/nuscenes"
# save_name = "samples_syntheocc_surocc"
# gt_path = os.path.join(dataroot, save_name)
# os.makedirs(gt_path, exist_ok=True)


# CAM_NAMES = [
#     "CAM_FRONT_LEFT",
#     "CAM_FRONT",
#     "CAM_FRONT_RIGHT",
#     "CAM_BACK_RIGHT",
#     "CAM_BACK",
#     "CAM_BACK_LEFT",
# ]
# for j in CAM_NAMES:
#     os.makedirs(gt_path + "/" + j, exist_ok=True)

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


def process_func(idx, rank):

    info = nus_pkl["infos"][idx]

    curr_name = info["lidar_path"].split("/")[-1]
    # occ_path = f"data/nuscenes/dense_voxels_with_semantic_z-5/{curr_name}.npy"
    occ_path = f"data/nuscenes/nuscenes_occ/samples/{curr_name}.npy"

    occ = np.load(occ_path)[:, [2, 1, 0, 3]]  # z y x c
    point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]

    num_classes = 16
    occupancy_size = [0.5, 0.5, 0.5]
    grid_size = [200, 200, 16]

    # occupancy_size = [0.2, 0.2, 0.2]
    # grid_size = [500, 500, 40]

    pc_range = torch.tensor(point_cloud_range)
    voxel_size = (pc_range[3:] - pc_range[:3]) / torch.tensor(grid_size)

    raw_w = 1600
    raw_h = 900

    # img_w = 100  # target reso
    # img_h = 56

    # img_w = 800
    # img_h = 448

    # img_w = 400  # target reso
    # img_h = 224

    img_w = 50  # target reso
    img_h = 28

    mtp_num = 96

    f = 0.0055

    def voxel2world(voxel):
        return voxel * voxel_size[None, :] + pc_range[:3][None, :]

    def world2voxel(wolrd):
        return (wolrd - pc_range[:3][None, :]) / voxel_size[None, :]

    colors_map = torch.tensor(
        [
            [0, 0, 0, 255],  # unknown
            [255, 158, 0, 255],  #  1 car  orange
            [255, 99, 71, 255],  #  2 truck  Tomato
            [255, 140, 0, 255],  #  3 trailer  Darkorange
            [255, 69, 0, 255],  #  4 bus  Orangered
            [233, 150, 70, 255],  #  5 construction_vehicle  Darksalmon
            [220, 20, 60, 255],  #  6 bicycle  Crimson
            [255, 61, 99, 255],  #  7 motorcycle  Red
            [0, 0, 230, 255],  #  8 pedestrian  Blue
            [47, 79, 79, 255],  #  9 traffic_cone  Darkslategrey
            [112, 128, 144, 255],  #  10 barrier  Slategrey
            [0, 207, 191, 255],  # 11  driveable_surface  nuTonomy green
            [175, 0, 75, 255],  #  12 other_flat
            [75, 0, 75, 255],  #  13  sidewalk
            [112, 180, 60, 255],  # 14 terrain
            [222, 184, 135, 255],  # 15 manmade Burlywood
            [0, 175, 0, 255],  # 16 vegetation  Green
        ]
    ).type(torch.uint8)


    c, r = np.meshgrid(np.arange(img_w), np.arange(img_h))
    uv = np.stack([c, r])
    uv = torch.tensor(uv)

    depth = (
        torch.arange(0.2, 51.4, 0.2)[..., None][..., None]
        .repeat(1, img_h, 1)
        .repeat(1, 1, img_w)
    )
    
    image_paths = []
    lidar2img_rts = []
    lidar2cam_rts = []
    cam_intrinsics = []
    cam_positions = []
    focal_positions = []
    for cam_type, cam_info in info["cams"].items():
        image_paths.append(cam_info["data_path"])
        cam_info["sensor2lidar_rotation"] = torch.tensor(
            cam_info["sensor2lidar_rotation"]
        )
        cam_info["sensor2lidar_translation"] = torch.tensor(
            cam_info["sensor2lidar_translation"]
        )
        cam_info["cam_intrinsic"] = torch.tensor(cam_info["cam_intrinsic"])
        # obtain lidar to image transformation matrix
        lidar2cam_r = torch.linalg.inv(cam_info["sensor2lidar_rotation"])
        lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
        lidar2cam_rt = torch.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        intrinsic = cam_info["cam_intrinsic"]
        viewpad = torch.eye(4)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        lidar2img_rt = viewpad @ lidar2cam_rt.T
        lidar2img_rts.append(lidar2img_rt)

        cam_intrinsics.append(viewpad)
        lidar2cam_rts.append(lidar2cam_rt.T)

        cam_position = torch.linalg.inv(lidar2cam_rt.T) @ torch.tensor(
            [0.0, 0.0, 0.0, 1.0]
        ).reshape([4, 1])
        cam_positions.append(cam_position.flatten()[:3])

        focal_position = torch.linalg.inv(lidar2cam_rt.T) @ torch.tensor(
            [0.0, 0.0, f, 1.0]
        ).reshape([4, 1])

        focal_positions.append(focal_position.flatten()[:3])

    occ = torch.tensor(occ)

    dense_vox = torch.zeros(grid_size).type(torch.uint8)
    occ_tr = occ[..., [2, 1, 0, 3]]

    dense_vox[occ_tr[:, 0], occ_tr[:, 1], occ_tr[:, 2]] = occ_tr[:, 3].type(torch.uint8)

    for cam_i in range(len(cam_intrinsics)):

        all_pcl = []
        all_col = []
        all_img_fov = []

        final_img = torch.zeros((img_h, img_w, 3)).type(torch.uint8)

        fuse_img = torch.zeros(
            (
                img_h,
                img_w,
            )
        ).type(torch.uint8)
        depth_map = torch.zeros((img_h, img_w, 1)).type(torch.uint8)

        curr_tr = lidar2cam_rts[cam_i]
        cam_in = cam_intrinsics[cam_i]
        c_u = cam_in[0, 2] / (raw_w / img_w)
        c_v = cam_in[1, 2] / (raw_h / img_h)
        f_u = cam_in[0, 0] / (raw_w / img_w)
        f_v = cam_in[1, 1] / (raw_h / img_h)

        b_x = cam_in[0, 3] / (-f_u)  # relative
        b_y = cam_in[1, 3] / (-f_v)

        dep_num = depth.shape[0]
        mtp_vis = []
        for _ in range(mtp_num):
            mtp_vis.append(
                torch.zeros(
                    (
                        img_h,
                        img_w,
                    )
                ).type(torch.uint8)
            )

        for dep_i in range(dep_num):
            # for dep_i in tqdm.tqdm(range(depth.shape[0])):
            dep_i = dep_num - 1 - dep_i

            uv_depth = (
                torch.cat([uv, depth[dep_i : dep_i + 1]], 0)
                .reshape((3, -1))
                .transpose(1, 0)
            )
            n = uv_depth.shape[0]
            x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
            y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
            pts_3d_rect = torch.zeros((n, 3))
            pts_3d_rect[:, 0] = x
            pts_3d_rect[:, 1] = y
            pts_3d_rect[:, 2] = uv_depth[:, 2]

            new_pcl = torch.cat([pts_3d_rect, torch.ones_like(pts_3d_rect[:, :1])], 1)

            new_pcl = torch.einsum("mn, an -> am", torch.linalg.inv(curr_tr), new_pcl)
            # new_pcl = torch.einsum("mn, an -> am", curr_tr, new_pcl)

            # new_pcl[:, :3] -= 0.1
            new_pcl[:, :3] -= occupancy_size[0] / 2

            new_pcl = world2voxel(new_pcl[:, :3])
            new_pcl = torch.round(new_pcl, decimals=0).type(torch.int32)

            pts_index = torch.zeros((new_pcl.shape[0])).type(torch.uint8)

            valid_flag = (
                ((new_pcl[:, 0] < grid_size[0]) & (new_pcl[:, 0] >= 0))
                & ((new_pcl[:, 1] < grid_size[1]) & (new_pcl[:, 1] >= 0))
                & ((new_pcl[:, 2] < grid_size[2]) & (new_pcl[:, 2] >= 0))
            )

            if valid_flag.max() > 0:
                pts_index[valid_flag] = dense_vox[
                    new_pcl[valid_flag][:, 0],
                    new_pcl[valid_flag][:, 1],
                    new_pcl[valid_flag][:, 2],
                ]

            col_pcl = torch.index_select(colors_map, 0, pts_index.type(torch.int32))

            img_fov = col_pcl[:, :3].reshape((img_h, img_w, 3))
            # cv2.imwrite(f"./exp/mtp/{dep_i:06d}.jpg", img_fov.cpu().numpy()[..., [2,1,0]])
            pts_index = pts_index.reshape(
                (
                    img_h,
                    img_w,
                )
            )
            img_flag = pts_index[..., None].repeat(1, 1, 3)
            final_img[img_flag != 0] = img_fov[img_flag != 0]

            all_img_fov.append(pts_index[None])

            mtp_idx = int(dep_i // (dep_num / mtp_num))
            mtp_vis[mtp_idx][pts_index != 0] = pts_index[pts_index != 0]
            fuse_img[pts_index != 0] = pts_index[pts_index != 0]

            depth_map[pts_index != 0] = dep_i

        save_path = image_paths[cam_i]

        if "samples" in save_path:
            save_path = save_path.replace("samples", save_name)
        if "sweeps" in save_path:
            save_path = save_path.replace(
                "sweeps", save_name.replace("samples", "sweeps")
            )

        final_img = final_img[..., [2, 1, 0]].cpu().numpy()
        cv2.imwrite(save_path[:-4] + "_occrgb.png", final_img)

        # rgb_img = cv2.imread(image_paths[cam_i])
        # rgb_img = cv2.resize(rgb_img, (img_w, img_h))
        # final_img = np.concatenate([final_img, rgb_img], 0)
        # # raw_occ_rgb = cv2.imread(save_path[:-4].replace(save_name, 'samples_syntheocc') + '_occrgb.jpg')
        # # final_img = np.concatenate([raw_occ_rgb, final_img], 0)
        # cv2.imwrite('output.jpg', final_img)

        if 1:
            all_img_fov = torch.cat(all_img_fov, 0).type(torch.uint8).flip(0)
            mtp_96 = torch.cat([x[None] for x in mtp_vis], 0).type(torch.uint8).flip(0)

            mtp_96_path = save_path[:-4] + "_mtp96.npz"
            mtp_256_path = save_path[:-4] + "_mtp256.npz"

            sparse_mat = mtp_96.cpu().numpy().reshape((-1, mtp_96.shape[-1]))
            # allmatrix_sp = sparse.coo_matrix(sparse_mat) # 采用行优先的方式压缩矩阵
            allmatrix_sp = sparse.csr_matrix(sparse_mat)  # 采用行优先的方式压缩矩阵
            sparse.save_npz(mtp_96_path, allmatrix_sp)  # 保存稀疏矩阵

            sparse_mat = all_img_fov.cpu().numpy().reshape((-1, all_img_fov.shape[-1]))
            # allmatrix_sp = sparse.coo_matrix(sparse_mat) # 采用行优先的方式压缩矩阵
            allmatrix_sp = sparse.csr_matrix(sparse_mat)  # 采用行优先的方式压缩矩阵
            sparse.save_npz(mtp_256_path, allmatrix_sp)  # 保存稀疏矩阵

            # allmatrix_sp = sparse.load_npz('allmatrix_sparse.npz')
            # allmatrix = allmatrix_sp.toarray().reshape(mtp_96.shape)

            fuse_path = save_path[:-4] + "_fuseweight.png"
            cv2.imwrite(fuse_path, fuse_img.cpu().numpy())

            depth_map_path = save_path[:-4] + "_depthmap.png"
            cv2.imwrite(depth_map_path, depth_map[..., 0].cpu().numpy())


def run_inference(rank, world_size, pred_results, input_datas):
    if rank is not None:
        # dist.init_process_group("gloo", rank=rank, world_size=world_size)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        rank = 0
    print(rank)

    torch.set_default_device(rank)

    all_list = input_datas[rank]  # [::6]

    for i in tqdm.tqdm(all_list):
        process_func(i, rank)


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


def get_multiview_imgs(scene_name, idx,return_len=None, offset=0, nusc_infos=None, scale_intrinsic=None, idx_permute_img=None, image_size=None):
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


if __name__ == "__main__":
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
    return_len = 1

    with open(imageset, 'rb') as f:
        data = pickle.load(f)

    nusc_infos = data['infos']
    scene_names = list(nusc_infos.keys())
    data_path = data_path
    scale_intrinsic = (img_size[0] / 1600, img_size[1] / 900)

    cur_origin_img_list = ['front', 'front_right', 'front_left', 'back', 'back_left', 'back_right']
    idx_permute_img = np.array([2, 0, 1, 5, 3, 4])
    view_order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]


    index = 0 # scene index
    idx = 0 # start time step 
    i = 0 # time step

    index = index % len(nusc_infos)
    scene_name = scene_names[index]

    token = nusc_infos[scene_name][idx + i]['token']
    label_file = os.path.join(data_path, f'{input_dataset}/{scene_name}/{token}/labels.npz')
    label = np.load(label_file)
    occ = label['semantics']

    # 576, 320
    images, all_camera_intrinsics, all_camera2ego = get_multiview_imgs(scene_name, idx,return_len=return_len, offset=0, nusc_infos=nusc_infos, scale_intrinsic=scale_intrinsic, idx_permute_img=idx_permute_img, image_size=img_size)

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
    K = torch.tensor(all_camera_intrinsics).permute(1, 0, 2, 3)
    T = torch.tensor(all_camera2ego).permute(1, 0, 2, 3)
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


    image_sem = Image.fromarray(img_fov[0].numpy())  
    image_sem.save('output_image_sem.png')  

    images[0].save('resize_image_rgb.png')


    

    all_multiview_imgs = []        
    for idx_frame in range(1):
        cur_multiview_imgs = []
        for idx_cam in idx_permute_img:
            cur_multiview_imgs.append(frames[idx_cam][idx_frame])
        cur_row_first = concat_images(cur_multiview_imgs[:3], pad=0)
        cur_row_last = concat_images(cur_multiview_imgs[3:], pad=0)
        cat_cur_multiview_imgs = concat_images([cur_row_first, cur_row_last], direction='vertical')

        all_multiview_imgs.append(cat_cur_multiview_imgs)

        cat_cur_multiview_imgs.save('{:06d}_{}.jpg'.format(idx, idx_frame))


    os.system("export NCCL_SOCKET_IFNAME=eth1")

    from torch.multiprocessing import Manager

    # world_size = 8
    world_size = 2

    all_len = len(nus_pkl["infos"])
    # val_len = all_len // 8 * 8
    val_len = all_len // world_size * world_size
    print(all_len, val_len)

    all_list = torch.arange(val_len).cpu().numpy()
    # all_list = torch.arange(16).cpu().numpy()

    # all_list = np.split(all_list, 8)
    all_list = np.split(all_list, world_size)

    input_datas = {}
    for i in range(world_size):
        input_datas[i] = list(all_list[i])
        print(len(input_datas[i]))

    input_datas[0] += list(np.arange(val_len, all_len))

    for i in range(world_size):
        print(len(input_datas[i]))

    # run_inference(0, 1, None, input_datas)
    # run_inference(None, 1, None, input_datas)

    with Manager() as manager:
        pred_results = manager.list()
        mp.spawn(
            run_inference,
            nprocs=world_size,
            args=(
                world_size,
                pred_results,
                input_datas,
            ),
            join=True,
        )
