import torch
import os
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
import imageio

from src.model.unet_spatio_temporal_condition_multiview import UNetSpatioTemporalConditionModelMultiview
from utils.custom_video_datasets import VideoNuscenesDataset
from torchvision import transforms
from src.pipeline.pipeline_stable_video_diffusion_multiview import StableVideoDiffusionPipelineMultiview

# Setting
# length = 14
# w = 1024
# h = 576

dataset_name = '../../../data/sample_nusc_video_all_cam_val.pkl'

length = 3
w = 576
h = 320
interval = 1
val_batch_size = 1
dataloader_num_workers = 2

# model path
pretrained_model_path = 'demo_model/img2video_1024_14f'
model_path = '../../work_dirs/nusc_fsdp_svd_front_576320_30f/checkpoint-0'

# initial image path
img_path = 'demo_img/0010_CameraFpgaP0H120.jpg'

output_folder = './output'
os.makedirs(output_folder,exist_ok=True)

# pipe = StableVideoDiffusionPipeline.from_pretrained(
#     model_path, torch_dtype=torch.float16, variant="fp16"
# )

pipe_param = {}
unet_path = os.path.join(model_path, 'unet')
unet = UNetSpatioTemporalConditionModelMultiview.from_pretrained(unet_path, torch_dtype=torch.float16)
unet.eval()
pipe_param['unet'] = unet

pipe = StableVideoDiffusionPipelineMultiview.from_pretrained(
    pretrained_model_path, **pipe_param, torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# image = load_image(img_path)

val_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

val_dataset = VideoNuscenesDataset(
    data_root=dataset_name,
    video_transforms=val_transforms,
    # tokenizer=tokenizer,
    video_length=length,
    interval = interval,
    img_size= (w, h),

    multi_view=True,

)

# DataLoaders creation:
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=val_batch_size,
    # batch_size=2,

    num_workers=dataloader_num_workers,
)


generator = torch.manual_seed(42)

for batch_dict in val_dataloader:

    batch_dict["pixel_values"] = batch_dict["pixel_values"].reshape(-1, *batch_dict["pixel_values"].shape[-4:])

    # frames = pipe(image, width=w, height=h,num_frames=length, num_inference_steps=25, noise_aug_strength=0.01, fps = 5, generator=generator).frames[0]
    frames = pipe(batch_dict, width=w, height=h,num_frames=length, num_inference_steps=25, noise_aug_strength=0, fps = 2, generator=generator).frames[0]

    export_path = os.path.join(output_folder, 'test.gif')

    # export to gif
    imageio.mimsave(export_path, frames, format='GIF', duration=200, loop=0)
