import torch
import os
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
import imageio

# from src.model.unet_spatio_temporal_condition_multiview import UNetSpatioTemporalConditionModelMultiview
from src.model.unet_spatio_temporal_condition_multiview_cond_dome import UNetSpatioTemporalConditionModelMultiviewCondDome
from utils.custom_video_datasets import VideoNuscenesDataset
from torchvision import transforms
# from src.pipeline.pipeline_stable_video_diffusion_multiview import StableVideoDiffusionPipelineMultiview
from src.pipeline.pipeline_stable_video_diffusion_multiview_cond_dome import StableVideoDiffusionPipelineMultiviewCondDome

from PIL import Image
import numpy as np

from mmengine import Config
from mmengine.registry import MODELS
from copy import deepcopy

from einops import rearrange


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

length = 2
w = 576
h = 320
interval = 1
val_batch_size = 1
dataloader_num_workers = 2
nframes_past = 1

# model path
pretrained_model_path = 'demo_model/img2video_1024_14f'
# model_path = '../../work_dirs/nusc_fsdp_svd_front_576320_30f/checkpoint-0'
model_path = '../../work_dirs/nusc_deepspeed_svd_front_576320_30f/checkpoint-0'

# initial image path
# img_path = 'demo_img/0010_CameraFpgaP0H120.jpg'

output_folder = './output'
os.makedirs(output_folder,exist_ok=True)

# pipe = StableVideoDiffusionPipeline.from_pretrained(
#     model_path, torch_dtype=torch.float16, variant="fp16"
# )

pipe_param = {}
unet_path = os.path.join(model_path, 'unet')
unet = UNetSpatioTemporalConditionModelMultiviewCondDome.from_pretrained(unet_path, torch_dtype=torch.float16)
unet.eval()
pipe_param['unet'] = unet

# pipe = StableVideoDiffusionPipelineMultiview.from_pretrained(
#     pretrained_model_path, **pipe_param, torch_dtype=torch.float16, variant="fp16"
# )
pipe = StableVideoDiffusionPipelineMultiviewCondDome.from_pretrained(
    pretrained_model_path, **pipe_param, torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# load config
cfg = Config.fromfile('utils/DOME/config/custom_train_dome.py')

import utils.DOME.model
from utils.DOME.dataset import get_dataloader, get_nuScenes_label_name

device = pipe._execution_device

occ_vae=MODELS.build(cfg.model.vae).to(device)
occ_vae.requires_grad_(False)
occ_vae.eval()

cfg.data_path = '../../../data/nuscenes/'
cfg.train_dataset_config.data_path = cfg.data_path
cfg.val_dataset_config.data_path = cfg.data_path
cfg.train_dataset_config.imageset = '../../../data/nuscenes_infos_train_temporal_v3_scene.pkl'
cfg.val_dataset_config.imageset = '../../../data/nuscenes_infos_val_temporal_v3_scene.pkl'
cfg.label_mapping = 'utils/DOME/config/label_mapping/nuscenes-occ.yaml'

train_dataloader, val_dataloader = get_dataloader(
    cfg.train_dataset_config,
    cfg.val_dataset_config,
    cfg.train_wrapper_config,
    cfg.val_wrapper_config,
    cfg.train_loader,
    cfg.val_loader)

# print(occ_vae.load_state_dict(torch.load('../../../ckpts/occvae_latest.pth')['state_dict']))
print(occ_vae.load_state_dict(torch.load('../../../ckpts/occvae_latest.pth')['state_dict'], strict=False))


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

# DataLoaders creation:
# val_dataloader = torch.utils.data.DataLoader(
#     val_dataset,
#     shuffle=False,
#     batch_size=val_batch_size,
#     # batch_size=2,

#     num_workers=dataloader_num_workers,
# )


generator = torch.manual_seed(42)

idx = 0

device = pipe._execution_device

with torch.no_grad():
    for batch_dict in val_dataloader:

        input_occs, target_occs, pixel_values, images, metas = batch_dict

        input_occs = input_occs.to(device)
        target_occs = target_occs.to(device)
        assert (input_occs==target_occs).all()

        bs,f,_,_,_=input_occs.shape
        encoded_latent, shape=occ_vae.forward_encoder(input_occs)
        encoded_latent,_,_=occ_vae.sample_z(encoded_latent) #bchw
        input_latents=encoded_latent*cfg.model.vae.scaling_factor

        # if input_latents.dim()==4:
        #     input_latents = rearrange(input_latents, '(b f) c h w -> b f c h w', b=bs).contiguous()
        # elif input_latents.dim()==5:
        #     input_latents = rearrange(input_latents, 'b c f h w -> b f c h w', b=bs).contiguous()
        # else:
        #     raise NotImplementedError
 
        occ_latent = input_latents.to(torch.float16)


        batch_dict = {"pixel_values": pixel_values.to(device), 'images': images}

        batch_dict["pixel_values"] = batch_dict["pixel_values"].reshape(-1, *batch_dict["pixel_values"].shape[-4:])
        
        # cond_frame_pixel_values = batch_dict["pixel_values"].to(torch.float16).reshape(-1, *batch_dict["pixel_values"].shape[-3:])
        # cond_frame_dist = pipe.vae.encode(cond_frame_pixel_values.to(device)).latent_dist
        # cond_frame = cond_frame_dist.sample()
        # cond_frame = cond_frame * pipe.vae.config.scaling_factor
        # cond_frame = cond_frame.reshape(-1, length, *cond_frame.shape[-3:])

        # initial_cond_mask = torch.zeros(cond_frame.shape[0], length, 1, 1, 1).to(torch.float16).to(device)
        # initial_cond_mask[:, :nframes_past] = 1

        # frames = pipe(image, width=w, height=h,num_frames=length, num_inference_steps=25, noise_aug_strength=0.01, fps = 5, generator=generator).frames[0]
        # frames = pipe(batch_dict, width=w, height=h,num_frames=length, num_inference_steps=25, noise_aug_strength=0, fps = 2, generator=generator).frames
        frames = pipe(batch_dict, width=w, height=h,num_frames=length, num_inference_steps=25, noise_aug_strength=0, fps = 2, generator=generator, nframes_past=nframes_past, occ_latent=occ_latent, metas=metas).frames

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


