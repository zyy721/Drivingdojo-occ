import torch
import os
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
import imageio

# from src.model.unet_spatio_temporal_condition_multiview import UNetSpatioTemporalConditionModelMultiview
from src.model.unet_spatio_temporal_condition_occ import UNetSpatioTemporalConditionModelOcc

from utils.custom_video_datasets import VideoNuscenesDataset
from torchvision import transforms
from src.pipeline.pipeline_stable_video_diffusion_occ import StableVideoDiffusionPipelineOcc

from PIL import Image
import numpy as np

from mmengine import Config
from mmengine.registry import MODELS
from copy import deepcopy


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

# front, front_right, back_right, back, back_left, front_left

origin_img_list = ['front', 'front_right', 'back_right', 'back', 'back_left', 'front_left']
idx_permute_img = [5, 0, 1, 2, 3, 4]
# permute_img_list = []
# for idx in idx_permute_img:
#     permute_img_list.append(origin_img_list[idx])

# Setting
# length = 14
# w = 1024
# h = 576

dataset_name = '../../../data/sample_nusc_video_all_cam_val.pkl'

# model path
pretrained_model_path = 'demo_model/img2video_1024_14f'
# model_path = '../../work_dirs/nusc_fsdp_svd_front_576320_30f/checkpoint-0'
model_path = '../../work_dirs/nusc_deepspeed_svd_occ_576320_30f/checkpoint-0'

# initial image path
# img_path = 'demo_img/0010_CameraFpgaP0H120.jpg'

output_folder = './output'
os.makedirs(output_folder,exist_ok=True)

# pipe = StableVideoDiffusionPipeline.from_pretrained(
#     model_path, torch_dtype=torch.float16, variant="fp16"
# )

pipe_param = {}
unet_path = os.path.join(model_path, 'unet')
# unet = UNetSpatioTemporalConditionModelMultiview.from_pretrained(unet_path, torch_dtype=torch.float16)
unet = UNetSpatioTemporalConditionModelOcc.from_pretrained(unet_path, torch_dtype=torch.float16)
unet.eval()
pipe_param['unet'] = unet

pipe = StableVideoDiffusionPipelineOcc.from_pretrained(
    pretrained_model_path, **pipe_param, torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# load config
# cfg = Config.fromfile('utils/OccWorld/config/custom_train_occworld.py')
cfg = Config.fromfile('utils/OccWorld/config/custom_occworld.py')
import utils.OccWorld.model
from utils.OccWorld.dataset import get_dataloader, get_nuScenes_label_name
from utils.OccWorld.utils.metric_util import MeanIoU, multi_step_MeanIou

start_frame=cfg.get('start_frame', 0)
mid_frame=cfg.get('mid_frame', 6) # 5
end_frame=cfg.get('end_frame', 12) # 11

length = 11
w = 576
h = 320
interval = 1
val_batch_size = 1
dataloader_num_workers = 2
nframes_past = 5

device = pipe._execution_device

occ_vae = MODELS.build(cfg.model)
occ_vae.init_weights()
occ_vae.requires_grad_(False)
occ_vae.to(device)

cfg.data_path = '../../../data/nuscenes/'
cfg.train_dataset_config.data_path = cfg.data_path
cfg.val_dataset_config.data_path = cfg.data_path
cfg.train_dataset_config.imageset = '../../../data/nuscenes_infos_train_temporal_v3_scene.pkl'
cfg.val_dataset_config.imageset = '../../../data/nuscenes_infos_val_temporal_v3_scene.pkl'
cfg.label_mapping = 'utils/OccWorld/config/label_mapping/nuscenes-occ.yaml'

train_dataloader, val_dataloader = get_dataloader(
    cfg.train_dataset_config,
    cfg.val_dataset_config,
    cfg.train_wrapper_config,
    cfg.val_wrapper_config,
    cfg.train_loader,
    cfg.val_loader)

ckpt_occ_vae = torch.load('demo_model/epoch_1.pth', map_location='cpu')
if 'state_dict' in ckpt_occ_vae:
    state_dict = ckpt_occ_vae['state_dict']
else:
    state_dict = ckpt_occ_vae
occ_vae.load_state_dict(state_dict, strict=True)

label_name = get_nuScenes_label_name(cfg.label_mapping)
unique_label = np.asarray(cfg.unique_label)
unique_label_str = [label_name[l] for l in unique_label]
CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg.get('ignore_label', -100), unique_label_str, 'sem', times=cfg.get('eval_length'))
CalMeanIou_vox = multi_step_MeanIou([1], cfg.get('ignore_label', -100), ['occupied'], 'vox', times=cfg.get('eval_length'))

occ_vae.eval()

CalMeanIou_sem.reset()
CalMeanIou_vox.reset()

# image = load_image(img_path)

# val_transforms = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]),
#     ]
# )

# val_dataset = VideoNuscenesDataset(
#     data_root=dataset_name,
#     video_transforms=val_transforms,
#     # tokenizer=tokenizer,
#     video_length=length,
#     interval = interval,
#     img_size= (w, h),

#     multi_view=True,

# )

# # DataLoaders creation:
# val_dataloader = torch.utils.data.DataLoader(
#     val_dataset,
#     shuffle=False,
#     batch_size=val_batch_size,
#     # batch_size=2,

#     num_workers=dataloader_num_workers,
# )

generator = torch.manual_seed(42)


idx = 0



for batch_dict in val_dataloader:

    # batch_dict["pixel_values"] = batch_dict["pixel_values"].reshape(-1, *batch_dict["pixel_values"].shape[-4:])
    
    # cond_frame_pixel_values = batch_dict["pixel_values"].to(torch.float16).reshape(-1, *batch_dict["pixel_values"].shape[-3:])
    # cond_frame_dist = pipe.vae.encode(cond_frame_pixel_values.to(device)).latent_dist
    # cond_frame = cond_frame_dist.sample()
    # cond_frame = cond_frame * pipe.vae.config.scaling_factor
    # cond_frame = cond_frame.reshape(-1, length, *cond_frame.shape[-3:])


    input_occs, target_occs, metas = batch_dict
    output_dict = {}
    output_dict['target_occs'] = input_occs[:, mid_frame:end_frame]

    batch_dict = dict()
    batch_dict["pixel_values"] = input_occs[:, :length]
    occ_z, occ_shapes = occ_vae.forward_encoder(input_occs[:, :length].to(device))
    latents, occ_z_mu, occ_z_sigma, occ_logvar = occ_vae.sample_z(occ_z)
    latents = latents.to(torch.float16)
    occ_z_mu = occ_z_mu.to(torch.float16)
    latents = latents.unsqueeze(0)
    occ_z_mu = occ_z_mu.unsqueeze(0)

    cond_frame = latents

    initial_cond_mask = torch.zeros(cond_frame.shape[0], length, 1, 1, 1).to(torch.float16).to(device)
    initial_cond_mask[:, :nframes_past] = 1

    # frames = pipe(image, width=w, height=h,num_frames=length, num_inference_steps=25, noise_aug_strength=0.01, fps = 5, generator=generator).frames[0]
    # frames = pipe(batch_dict, width=w, height=h,num_frames=length, num_inference_steps=25, noise_aug_strength=0, fps = 2, generator=generator).frames
    frames = pipe(batch_dict, width=w, height=h,num_frames=length, num_inference_steps=25, noise_aug_strength=0, fps = 2, generator=generator, cond_mask=initial_cond_mask, nframes_past=nframes_past, cond_frame=cond_frame, occ_z_mu=occ_z_mu, occ_vae=occ_vae).frames
    frames = frames[mid_frame:].to(torch.float16)

    z_q_predict = occ_vae.forward_decoder(frames, occ_shapes, output_dict['target_occs'].shape)
    output_dict['logits'] = z_q_predict
    pred = z_q_predict.argmax(dim=-1).detach().cuda()
    output_dict['sem_pred'] = pred
    pred_iou = deepcopy(pred)
    
    pred_iou[pred_iou!=17] = 1
    pred_iou[pred_iou==17] = 0
    output_dict['iou_pred'] = pred_iou

    result_dict = output_dict

    if result_dict.get('target_occs', None) is not None:
        target_occs = result_dict['target_occs']
    target_occs_iou = deepcopy(target_occs)
    target_occs_iou[target_occs_iou != 17] = 1
    target_occs_iou[target_occs_iou == 17] = 0

    CalMeanIou_sem._after_step(result_dict['sem_pred'], target_occs)
    CalMeanIou_vox._after_step(result_dict['iou_pred'], target_occs_iou)

    # export_path = os.path.join(output_folder, 'test.gif')

    # # export to gif
    # imageio.mimsave(export_path, frames, format='GIF', duration=200, loop=0)

    all_multiview_imgs = []        
    for idx_frame in range(length):
        cur_multiview_imgs = []
        for idx_cam in idx_permute_img:
            cur_multiview_imgs.append(frames[idx_cam][idx_frame])
        cur_row_first = concat_images(cur_multiview_imgs[:3], pad=0)
        cur_row_last = concat_images(cur_multiview_imgs[3:], pad=0)
        cat_cur_multiview_imgs = concat_images([cur_row_first, cur_row_last], direction='vertical')

        all_multiview_imgs.append(cat_cur_multiview_imgs)

        cat_cur_multiview_imgs.save('{:06d}_{}.jpg'.format(idx, idx_frame))

    imageio.mimsave(os.path.join(output_folder, '{:06d}.mp4'.format(idx)), all_multiview_imgs, fps=2)
    idx += 1


