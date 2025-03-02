import os, numpy as np, pickle
from pyquaternion import Quaternion
from copy import deepcopy
try:
    from . import OPENOCC_DATASET
except:
    from mmengine.registry import Registry
    OPENOCC_DATASET = Registry('openocc_dataset')
import torch

# from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, Box3DMode
from pathlib import Path
from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation
import re


from torchvision import transforms
from torchvision import transforms as tr
# from examples.world_model.utils.image_datasets import default_loader

try:
    from examples.world_model.utils.image_datasets import default_loader
except:
    from utils.image_datasets import default_loader


@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar:
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            times=5,
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts',
            new_rel_pose=False,
            test_index_offset=0
        ):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        self.nusc_infos = data['infos']  #TODO
        # self.nusc_infos = dict(list(data['infos'].items())[::50])# data['infos']  #debug #TODO
        # self.nusc_infos = dict(list(data['infos'].items())[::10])# data['infos']  #debug
        # self.nusc_infos = dict(list(data['infos'].items())[:1])# data['infos']  #debug
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.nusc = nusc
        self.times = times
        self.test_mode = test_mode
        # assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        # assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset
        self.new_rel_pose=new_rel_pose
        self.test_index_offset=test_index_offset


        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        ) 
        img_size = (576, 320)

        self.loader = default_loader
        self.img_transform = train_transforms
        self.image_size = img_size
        self.default_transforms = tr.Compose(
            [
                tr.ToTensor(),
            ]
        )

        self.scale_intrinsic = (img_size[0] / 1600, img_size[1] / 900)
        # self.origin_img_list = ['front', 'front_right', 'back_right', 'back', 'back_left', 'front_left']
        # self.idx_permute_img = [5, 0, 1, 2, 3, 4]

        self.cur_origin_img_list = ['front', 'front_right', 'front_left', 'back', 'back_left', 'back_right']
        self.idx_permute_img = np.array([2, 0, 1, 5, 3, 4])
        self.view_order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]

        # permute_img_list = []
        # for idx in self.idx_permute_img:
        #     permute_img_list.append(self.cur_origin_img_list[idx])

        self.train = True
        if 'val' in imageset:
            self.train = False


    def load_and_transform_frames(self, frame_list, loader, img_transform=None, img_size=(576,320)):
        assert(isinstance(frame_list, list)), "frame_list must be a list not {}".format(type(frame_list))
        clip = []

        for frame in frame_list:
            if isinstance(frame, tuple):
                fpath, label = frame
            elif isinstance(frame, dict):
                fpath = frame["img_path"]
            else:
                fpath = frame
            
            img = loader(fpath)
            img = img.resize(img_size)
            if img_transform is not None:
                img = img_transform(img)
            else:
                img = self.default_transforms(img)
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            clip.append(img)
        return clip


        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)*self.times

    def __getitem__(self, index):
        index = index % len(self.nusc_infos)
        scene_name = self.scene_names[index]
        scene_len = self.scene_lens[index]
        return_len_=min(self.return_len,scene_len- self.offset-self.test_index_offset)
        if not self.test_mode:
            # idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
            idx = np.random.randint(0, scene_len - return_len_ - self.offset + 1)
            # print('@'*50,index,idx,scene_len - self.return_len - self.offset + 1,len(self.scene_names),self.scene_names[0:5],self.scene_names[-5:])
        else:
            # idx=0
            idx=self.test_index_offset
            # print('@'*10,idx)
        occs = []
        for i in range(return_len_ + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        input_occs = np.stack(occs, dtype=np.int64)
        occs = []
        for i in range(return_len_ + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        output_occs = np.stack(occs, dtype=np.int64)
        metas = {}
        metas.update(scene_token=self.nusc_infos[scene_name][4]['token'])
        metas.update(self.get_meta_data(scene_name, idx,return_len=return_len_))
        # return input_occs[:self.return_len], output_occs[self.offset:], metas


        pixel_values, images, all_camera_intrinsics, all_camera2ego = self.get_multiview_imgs(scene_name, idx,return_len=return_len_)
        # metas.update(images=images)

        metas.update(new_cam_intrinsic=all_camera_intrinsics, new_cam2ego=all_camera2ego)

        if not self.train:
            metas.update(index=index, idx=idx)

        return input_occs[:self.return_len], output_occs[self.offset:], pixel_values, images, metas


    def get_multiview_imgs(self, scene_name, idx,return_len=None):
        pixel_values_list = []
        images_list = []
        cam_front_list, cam_front_right_list, cam_front_left_list, cam_back_list, cam_back_left_list, cam_back_right_list = [], [], [], [], [], []
        all_camera2ego_list, all_camera_intrinsics_list = [], []
        # front, front_right, front_left, back, back_left, back_right
        for i in range(return_len + self.offset):
            cur_nusc_info = self.nusc_infos[scene_name][idx+i]
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

        all_camera_intrinsics[:, :, 0, 0] = self.scale_intrinsic[0] * all_camera_intrinsics[:, :, 0, 0]
        all_camera_intrinsics[:, :, 0, 2] = self.scale_intrinsic[0] * all_camera_intrinsics[:, :, 0, 2]
        all_camera_intrinsics[:, :, 1, 1] = self.scale_intrinsic[1] * all_camera_intrinsics[:, :, 1, 1]
        all_camera_intrinsics[:, :, 1, 2] = self.scale_intrinsic[1] * all_camera_intrinsics[:, :, 1, 2]

        all_camera_intrinsics = np.transpose(all_camera_intrinsics, axes=(1, 0, 2, 3))
        all_camera2ego = np.transpose(all_camera2ego, axes=(1, 0, 2, 3))

        all_camera_intrinsics = all_camera_intrinsics[self.idx_permute_img]
        all_camera2ego = all_camera2ego[self.idx_permute_img]

        if self.train:
            for i in range(return_len + self.offset):
                cur_nusc_info = self.nusc_infos[scene_name][idx+i]
                cam_front_list.append('.' + cur_nusc_info['cams']['CAM_FRONT']['data_path'])
                cam_front_right_list.append('.' + cur_nusc_info['cams']['CAM_FRONT_RIGHT']['data_path'])
                cam_front_left_list.append('.' + cur_nusc_info['cams']['CAM_FRONT_LEFT']['data_path'])
                cam_back_list.append('.' + cur_nusc_info['cams']['CAM_BACK']['data_path'])
                cam_back_left_list.append('.' + cur_nusc_info['cams']['CAM_BACK_LEFT']['data_path'])
                cam_back_right_list.append('.' + cur_nusc_info['cams']['CAM_BACK_RIGHT']['data_path'])
        else:
            for i in range(return_len + self.offset):
                cur_nusc_info = self.nusc_infos[scene_name][idx+i]
                cam_front_list.append('../../.' + cur_nusc_info['cams']['CAM_FRONT']['data_path'])
                cam_front_right_list.append('../../.' + cur_nusc_info['cams']['CAM_FRONT_RIGHT']['data_path'])
                cam_front_left_list.append('../../.' + cur_nusc_info['cams']['CAM_FRONT_LEFT']['data_path'])
                cam_back_list.append('../../.' + cur_nusc_info['cams']['CAM_BACK']['data_path'])
                cam_back_left_list.append('../../.' + cur_nusc_info['cams']['CAM_BACK_LEFT']['data_path'])
                cam_back_right_list.append('../../.' + cur_nusc_info['cams']['CAM_BACK_RIGHT']['data_path'])

        all_cam_dict = {"CAM_FRONT_LEFT": cam_front_left_list, "CAM_FRONT": cam_front_list, "CAM_FRONT_RIGHT": cam_front_right_list, 
                        "CAM_BACK_RIGHT": cam_back_right_list, "CAM_BACK": cam_back_list, "CAM_BACK_LEFT": cam_back_left_list}
        for cam_name in self.view_order:
            video = all_cam_dict[cam_name]
            frames = self.load_and_transform_frames(video, self.loader, self.img_transform, img_size=self.image_size)

            frames = torch.cat(frames, 1) # c,t,h,w
            frames = frames.transpose(0, 1) # t,c,h,w

            # frames_low = self.load_and_transform_frames(video, self.loader, self.img_transform, img_size=(288,160))
            # frames_low = torch.cat(frames_low, 1) # c,t,h,w
            # frames_low = frames_low.transpose(0, 1) # t,c,h,w
            
            pixel_values_list.append(frames)
            images_list.append(self.load_and_transform_frames(video, self.loader, img_size=self.image_size))

        pixel_values = torch.stack(pixel_values_list)
        images = images_list

        return pixel_values, images, all_camera_intrinsics, all_camera2ego


    def get_meta_data(self, scene_name, idx,return_len=None):
        gt_modes = []
        xys = []
        e2g_t=[]
        e2g_r=[]
        return_len=self.return_len if return_len is None else return_len

        for i in range(return_len + self.offset):
            xys.append(self.nusc_infos[scene_name][idx+i]['gt_ego_fut_trajs'][0]) #1*2 #[array([-0.0050938,  3.8259335], dtype=float32)]
            e2g_t.append(self.nusc_infos[scene_name][idx+i]['ego2global_translation'])
            e2g_r.append(self.nusc_infos[scene_name][idx+i]['ego2global_rotation'])
            gt_modes.append(self.nusc_infos[scene_name][idx+i]['pose_mode'])# [0,0,1] #maybe type selection bewteen (angle,speed,trajectory) #may 直行左右转
        xys = np.asarray(xys)
        gt_modes = np.asarray(gt_modes)
        e2g_t=np.array(e2g_t)
        e2g_r=np.array(e2g_r)
        # use max mode as the whole traj mode 0: right 1: left 2:straight
        traj_mode=np.argmax(gt_modes.sum(0)).item()

        #get traj (not velocity)  relative to first frame
        e2g_rel0_t=e2g_t.copy()
        e2g_rel0_r=e2g_r.copy()
        for i in range(return_len + self.offset):
            r0=Quaternion(e2g_r[0]).rotation_matrix
            ri=Quaternion(e2g_r[i]).rotation_matrix
            e2g_rel0_t[i]=np.linalg.inv(r0)@(e2g_t[i]-e2g_t[0])
            e2g_rel0_r[i]=Quaternion(matrix=np.linalg.inv(r0)@ri).elements
        poses=[]
        for tt,rr in zip(e2g_t,e2g_r):
            pose=np.eye(4)
            pose[:3,3]=tt
            pose[:3,:3]=Quaternion(rr).rotation_matrix
            poses.append(pose)
        poses=np.stack(poses,axis=0)

        meta_data2=get_meta_data(poses)

        rel_poses_yaws=meta_data2['rel_poses_yaws']
        if self.new_rel_pose:
            xys=meta_data2['rel_poses']
        
        return {'rel_poses': xys, 'gt_mode': gt_modes, 'e2g_t':e2g_t,'e2g_r':e2g_r,'traj_mode':traj_mode,
                'e2g_rel0_t':e2g_rel0_t,'e2g_rel0_r':e2g_rel0_r,
                'rel_poses_yaws':rel_poses_yaws,
        }

    def get_traj_mode(self, scene_name, idx):
        gt_modes = []
        for i in range(self.return_len + self.offset):
            gt_modes.append(self.nusc_infos[scene_name][idx+i]['pose_mode'])# [0,0,1] #maybe type selection bewteen (angle,speed,trajectory) #may 直行左右转
        gt_modes = np.asarray(gt_modes)
        # use max mode as the whole traj mode 0: right 1: left 2:straight
        traj_mode=np.argmax(gt_modes.sum(0)).item()
        return traj_mode

    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []
        
        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t
            
            
            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            #cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            #focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
        
        
        
        
        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))
        
        return input_dict
        
@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidarTraverse(nuScenesSceneDatasetLidar):
    def __init__(
        self,
        data_path,
        return_len,
        offset,
        imageset='train',
        nusc=None,
        times=1,
        test_mode=False,
        use_valid_flag=True,
        input_dataset='gts',
        output_dataset='gts',
        **kwargs
    ):
        super().__init__(data_path, return_len, offset, imageset, nusc, times, test_mode, input_dataset, output_dataset,**kwargs)
        self.scene_lens = [l - self.return_len - self.offset for l in self.scene_lens]
        # self.scene_lens=self.scene_lens[:1]        #debug
        self.use_valid_flag = use_valid_flag
        self.CLASSES = [
            'noise', 'animal' ,'human.pedestrian.adult', 'human.pedestrian.child',
            'human.pedestrian.construction_worker',
            'human.pedestrian.personal_mobility',
            'human.pedestrian.police_officer',
            'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
            'movable_object.barrier', 'movable_object.debris',
            'movable_object.pushable_pullable', 'movable_object.trafficcone',
            'static_object.bicycle_rack', 'vehicle.bicycle',
            'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car',
            'vehicle.construction', 'vehicle.emergency.ambulance',
            'vehicle.emergency.police', 'vehicle.motorcycle',
            'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
            'flat.other', 'flat.sidewalk', 'flat.terrain', 'flat.traffic_marking',
            'static.manmade', 'static.other', 'static.vegetation',
            'vehicle.ego'
        ]
        self.with_velocity = True
        self.with_attr = True
        # self.box_mode_3d = Box3DMode.LIDAR
        
    def __len__(self):
        'Denotes the total number of samples'
        return sum(self.scene_lens)
    
    def __getitem__(self, index):
        for i, scene_len in enumerate(self.scene_lens):
            if index < scene_len:
                scene_name = self.scene_names[i]
                idx = index
                break
            else:
                index -= scene_len
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        input_occs = np.stack(occs, dtype=np.int64)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        output_occs = np.stack(occs, dtype=np.int64)
        metas = {}
        metas.update(scene_name=scene_name)
        metas.update(scene_token=self.nusc_infos[scene_name][4]['token'])
        metas.update(self.get_meta_data(scene_name, idx))
        # metas.update(self.get_ego_action_info(scene_name,idx))
        # if self.test_mode:
            # metas.update(self.get_meta_info(scene_name, idx))
        # metas.update(self.get_image_info(scene_name,idx))
        # import pdb; pdb.set_trace()
        return input_occs[:self.return_len], output_occs[self.offset:], metas
    
    def get_ego_action_info(self, scene_name, idx):
        vels=[]
        steers=[]
        accels=[]
        for i in range(self.return_len + self.offset):
            accel = np.linalg.norm(self.nusc_infos[scene_name][idx + i]['can_bus'][7:10]) #todo direction
            vel = np.linalg.norm(self.nusc_infos[scene_name][idx + i]['can_bus'][13: 16])
            steer = self.nusc_infos[scene_name][idx + i]['can_bus'][16]
            accels.append(accel)
            vels.append(vel)
            steers.append(steer)
        accels = np.array(accels)
        vels = np.array(vels)
        steers = np.array(steers)
        return {'vels': vels,'steers': steers, 'accels': accels}

    def get_meta_info(self, scene_name, idx):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        fut_valid_flag = info['valid_flag']
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        '''gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
                print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        '''
        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        if self.with_attr:
            gt_fut_trajs = info['gt_agent_fut_trajs'][mask]
            gt_fut_masks = info['gt_agent_fut_masks'][mask]
            gt_fut_goal = info['gt_agent_fut_goal'][mask]
            gt_lcf_feat = info['gt_agent_lcf_feat'][mask]
            gt_fut_yaw = info['gt_agent_fut_yaw'][mask]
            attr_labels = np.concatenate(
                [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
            ).astype(np.float32)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # gt_bboxes_3d = LiDARInstance3DBoxes(
        #     gt_bboxes_3d,
        #     box_dim=gt_bboxes_3d.shape[-1],
        #     origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        # anns_results = dict(
        #     gt_bboxes_3d=gt_bboxes_3d,
        #     #gt_labels_3d=gt_labels_3d,
        #     gt_names=gt_names_3d,
        #     attr_labels=attr_labels,
        #     fut_valid_flag=fut_valid_flag,)
        
        anns_results = dict(
            # gt_bboxes_3d=gt_bboxes_3d,
            # #gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            attr_labels=attr_labels,
            fut_valid_flag=fut_valid_flag,)

        return anns_results
        
        
        
    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []
        
        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t
            
            
            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            #cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            #focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
        
        
        
        
        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))
        
        return input_dict


@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidarResample:
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            times=1,
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts',
            raw_times=1,
            resample_times=1,
        ):

        self.scene_lens ,self.nusc_infos = self.load_data(data_path,raw_times,resample_times)
        # self.nusc_infos = self.nusc_infos[::10]# data['infos']  #debug #TODO
        # self.scene_lens = self.scene_lens[::10]   # data['infos']  #debug #TODO
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.nusc = nusc
        self.times = times
        self.test_mode = test_mode
    
    def load_data(self, data_path,raw_times=0,resample_times=1):
        cache_path=f'{data_path}/scene_cache.npz'
        if os.path.exists(cache_path):
            data = np.load(cache_path,allow_pickle=True)
            # return data['all_scene_lens'], data['all_occs_path']
            all_scene_lens=data['all_scene_lens'].tolist()
            all_occs_path=data['all_occs_path'].tolist()
            all_scene_lens_raw=data['all_scene_lens_raw'].tolist()
            all_occs_path_raw=data['all_occs_path_raw'].tolist()
        else:
            def process_scene(src_scene,scene_key='scene*'):
                scene_lens = []
                occs_path = []
                for resample_scene in sorted(src_scene.glob(scene_key)):
                    all_traj = list(sorted(resample_scene.glob('traj*')))
                    scene_lens.append(len(all_traj))
                    occs_path_i = [(traj/'labels.npz').as_posix() for traj in all_traj]
                    occs_path.append(occs_path_i)
                return scene_lens, occs_path

            all_src_scenes=sorted(list(Path(data_path).glob('src_scene*')))
            total_scenes = len(all_src_scenes)
            
            results = Parallel(n_jobs=-1)(
                delayed(process_scene)(src_scene,scene_key='scene*') 
                for src_scene in tqdm(all_src_scenes, total=total_scenes, desc="Processing scenes")
            )
            
            all_scene_lens = []
            all_occs_path = []
            for scene_lens, occs_path in results:
                all_scene_lens.extend(scene_lens)
                all_occs_path.extend(occs_path)

            # add raw_scene
            results = Parallel(n_jobs=-1)(
                delayed(process_scene)(src_scene,scene_key='raw_scene') 
                for src_scene in tqdm(all_src_scenes, total=total_scenes, desc="Processing scenes")
            )
            
            all_scene_lens_raw = []
            all_occs_path_raw = []
            for scene_lens, occs_path in results:
                all_scene_lens_raw.extend(scene_lens)
                all_occs_path_raw.extend(occs_path)

            np.savez(cache_path, all_scene_lens=all_scene_lens,all_occs_path=all_occs_path,all_scene_lens_raw=all_scene_lens_raw,all_occs_path_raw=np.array(all_occs_path_raw, dtype="object"))

        all_scene_lens=all_scene_lens*resample_times+all_scene_lens_raw*raw_times
        all_occs_path=all_occs_path*resample_times+all_occs_path_raw*raw_times

        return all_scene_lens, all_occs_path
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)*self.times

    def __getitem__(self, index):
        scene_index = index % len(self.nusc_infos)
        scene_len = self.scene_lens[scene_index]
        if not self.test_mode:
            idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        else:
            idx=0
        occs = []
        poses=[]
        for i in range(self.return_len + self.offset):
            iidx=idx + i
            # if iidx>=scene_len:
            #     iidx=scene_len-1
            #     print(f'warning: {iidx} out of range, scene_len: {scene_len}')
            label_file = self.nusc_infos[scene_index][iidx]
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
            poses.append(label['pose'])
        input_occs = np.stack(occs, dtype=np.int64)
        poses=np.stack(poses, dtype=np.float32)
        output_occs = input_occs.copy()
        metas = {}
        metas.update(get_meta_data(poses))
        metas['src_scenes']=int(re.search(r'src_scene-(\d{4})', label_file).group(1)) #TODO might bug
        return input_occs[:self.return_len], output_occs[self.offset:], metas

def get_meta_data(poses):
    rel_pose = np.linalg.inv(poses[:-1]) @ poses[1:]
    rel_pose=  np.concatenate([rel_pose,rel_pose[-1:]], axis=0)
    xyzs = rel_pose[:, :3, 3]

    xys = xyzs[:, :2]
    e2g_t = poses[:, :3, 3]
    # rot 2 quat
    e2g_r = np.array([Quaternion(matrix=pose[:3, :3],atol=1e-7).elements for pose in poses])
    rel_yaws = Rotation.from_matrix(rel_pose[:,:3,:3]).as_euler('zyx', degrees=False)[:,0]

    #get traj (not velocity)  relative to first frame
    e2g_rel0_t = e2g_t.copy()
    # Convert rotations to rotation matrices
    e2g_r_w_last = e2g_r.copy()
    e2g_r_w_last[:, [0, 1, 2 ,3]] = e2g_r_w_last[:, [1, 2,3, 0]] 
    r0 = Rotation.from_quat(e2g_r_w_last[0]).as_matrix()  # First rotation matrix
    rotations = Rotation.from_quat(e2g_r_w_last).as_matrix()  # All rotation matrices
    e2g_rel0_t = np.linalg.inv(r0) @ ( e2g_t - e2g_t[0]).T
    e2g_rel0_t = e2g_rel0_t.T

    rr=np.array([
        [0,-1],
        [1,0],]
    )
    xys=xys@rr.T
    rel_poses_yaws=np.concatenate([xys,rel_yaws[:,None]],axis=1)
    
    return {
        'rel_poses': xys,
        'rel_poses_xyz': xyzs,
        'e2g_t': e2g_t,
        'e2g_r': e2g_r,
        'rel_poses_yaws':rel_poses_yaws,
        'e2g_rel0_t':e2g_rel0_t
    }

