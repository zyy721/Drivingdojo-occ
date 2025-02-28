from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

# from ...configuration_utils import ConfigMixin, register_to_config
# from ...loaders import UNet2DConditionLoadersMixin
# from ...utils import BaseOutput, logging
# from ..attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
# from ..embeddings import TimestepEmbedding, Timesteps
# from ..modeling_utils import ModelMixin
# from .unet_3d_blocks import UNetMidBlockSpatioTemporal, get_down_block, get_up_block

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging

from diffusers.models.unets.unet_spatio_temporal_condition import (
    UNetSpatioTemporalConditionModel,
    UNetSpatioTemporalConditionOutput,
)

from diffusers.models.attention import BasicTransformerBlock, TemporalBasicTransformerBlock

from ..misc.common import _get_module, _set_module
from .blocks import (
    BasicMultiviewTransformerBlock, TemporalBasicMultiviewTransformerBlock
)

from diffusers.models.embeddings import TimestepEmbedding, Timesteps


from utils.cldm.volume_transform import VolumeTransform
import torch.nn.functional as F
from einops import rearrange
from utils.DOME.model.VAE.vae_2d_resnet import ResnetBlock, Upsample
from diffusers.models.controlnet import zero_module


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# @dataclass
# class UNetSpatioTemporalConditionOutput(BaseOutput):
#     """
#     The output of [`UNetSpatioTemporalConditionModel`].

#     Args:
#         sample (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
#             The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
#     """

#     sample: torch.Tensor = None


# class UNetSpatioTemporalConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
class UNetSpatioTemporalConditionModelMultiviewCondDome(UNetSpatioTemporalConditionModel):

    r"""
    A conditional Spatio-Temporal UNet model that takes a noisy video frames, conditional state, and a timestep and
    returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 8): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "DownBlockSpatioTemporal")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        addition_time_embed_dim: (`int`, defaults to 256):
            Dimension to to encode the additional time ids.
        projection_class_embeddings_input_dim (`int`, defaults to 768):
            The dimension of the projection of encoded `added_time_ids`.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int`, `Tuple[int]`, or `Tuple[Tuple]` , *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unets.unet_3d_blocks.CrossAttnDownBlockSpatioTemporal`],
            [`~models.unets.unet_3d_blocks.CrossAttnUpBlockSpatioTemporal`],
            [`~models.unets.unet_3d_blocks.UNetMidBlockSpatioTemporal`].
        num_attention_heads (`int`, `Tuple[int]`, defaults to `(5, 10, 10, 20)`):
            The number of attention heads.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types: Tuple[str] = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 20, 20),
        num_frames: int = 25,
        # parameter added, we should keep all above (do not use kwargs)
        trainable_state="only_new",
        neighboring_view_pair: Optional[dict] = None,
        neighboring_attn_type: str = "add",
        zero_module_type: str = "zero_linear",
        crossview_attn_type: str = "basic",
        img_size: Optional[Tuple[int, int]] = None,

        # nframes_past: int = 10,
        cur_num_frames: int = 10,

    ):
        
        super().__init__(
            sample_size=sample_size, in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            addition_time_embed_dim=addition_time_embed_dim,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            num_attention_heads=num_attention_heads,
            num_frames=num_frames,)

        # self.custom_conv_in = nn.Conv2d(
        #     4 * (nframes_past + 1),
        #     block_out_channels[0],
        #     kernel_size=3,
        #     padding=1,
        # )

        # time_embed_dim = block_out_channels[0] * 4
        # timestep_input_dim = block_out_channels[0]
        # self.cond_time_stack_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.crossview_attn_type = crossview_attn_type
        self.img_size = [int(s) for s in img_size] \
            if img_size is not None else None
        self._new_module = {}
        for name, mod in list(self.named_modules()):
            if isinstance(mod, BasicTransformerBlock):
                if crossview_attn_type == "basic":
                    _set_module(self, name, BasicMultiviewTransformerBlock(
                        **mod._args,
                        neighboring_view_pair=neighboring_view_pair,
                        neighboring_attn_type=neighboring_attn_type,
                        zero_module_type=zero_module_type,
                        cur_num_frames=cur_num_frames,
                    ))
                else:
                    raise TypeError(f"Unknown attn type: {crossview_attn_type}")
                for k, v in _get_module(self, name).new_module.items():
                    self._new_module[f"{name}.{k}"] = v
            if isinstance(mod, TemporalBasicTransformerBlock):
                if crossview_attn_type == "basic":
                    _set_module(self, name, TemporalBasicMultiviewTransformerBlock(
                        **mod._args,
                        neighboring_view_pair=neighboring_view_pair,
                        neighboring_attn_type=neighboring_attn_type,
                        zero_module_type=zero_module_type,
                        cur_num_frames=cur_num_frames,
                    ))
                else:
                    raise TypeError(f"Unknown attn type: {crossview_attn_type}")
                for k, v in _get_module(self, name).new_module.items():
                    self._new_module[f"{name}.{k}"] = v
        self.trainable_state = trainable_state


        origin_occ_shape = (16, 200, 200)
        input_size = (320, 576)
        down_sample = 8
        grid_config = dict(
            x_bound = [-40.0, 40.0, 0.4],
            y_bound = [-40.0, 40.0, 0.4],
            z_bound = [-1.0, 5.4, 0.4],
            d_bound = [0.5, 48.5, 1.0],
        )

        self.ProjResnetBlock_0 = ResnetBlock(in_channels=64,
                                           out_channels=64,
                                           temb_channels=0,
                                           dropout=0)
        self.ProjUpsample = Upsample(64, with_conv=True)

        self.VT = VolumeTransform(with_DSE=True, 
                                  origin_occ_shape=origin_occ_shape, 
                                  input_size=input_size,
                                  down_sample=down_sample,
                                  grid_config=grid_config,
                                  )

        self.ProjResnetBlock_1 = ResnetBlock(in_channels=16,
                                           out_channels=16,
                                           temb_channels=0,
                                           dropout=0)
        
        self.connector = zero_module(torch.nn.Conv2d(in_channels=16,
                                     out_channels=8,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1))

        self.cur_num_frames = cur_num_frames
        
    def convert_3d_to_2d(
        self,
        occ_latent,
        metas,
    ):
        # [2, 64, 2, 25, 25]
        occ_latent = rearrange(occ_latent, 'b c f h w -> (b f) c h w').contiguous()

        volume_feats = self.ProjResnetBlock_0(occ_latent)
        volume_feats = self.ProjUpsample(volume_feats, shape=[50, 50])

        volume_feats = rearrange(volume_feats, 'b (d c_new) h w -> b c_new d h w', d=4)

        # [(2 6), 2, 4, 4]
        padcam2ego = metas[0]['new_cam2ego']
        cam_intrinsic = metas[0]['new_cam_intrinsic']
        
        # padcam2ego = torch.tensor(padcam2ego).permute(1, 0, 2, 3).to(volume_feats.device)
        # cam_intrinsic = torch.tensor(cam_intrinsic).permute(1, 0, 2, 3).to(volume_feats.device)

        padcam2ego = torch.tensor(padcam2ego).to(volume_feats.device)
        cam_intrinsic = torch.tensor(cam_intrinsic).to(volume_feats.device)

        padcam2ego = rearrange(padcam2ego, '(b n_cam) f h w -> (b f) n_cam h w', n_cam=6)
        cam_intrinsic = rearrange(cam_intrinsic, '(b n_cam) f h w -> (b f) n_cam h w', n_cam=6)

        T = padcam2ego
        K = cam_intrinsic

        T = T.reshape(-1, T.shape[2], T.shape[3])  # B,3,4,4
        K = K.reshape(-1, K.shape[2], K.shape[3])  # B*3,3,3
        volume_feats = volume_feats.unsqueeze(1).repeat(1,6,1,1,1,1)\
                    .reshape(-1, volume_feats.shape[1], volume_feats.shape[2], volume_feats.shape[3], volume_feats.shape[4])  # B*3,C,H,W,D
        volume_feats = self.VT(volume_feats, K, T)  # B*6,C,D,H,W

        volume_feats = self.ProjResnetBlock_1(volume_feats)
        volume_feats = self.connector(volume_feats)

        # volume_feats = rearrange(volume_feats, '(f n_cam) c h w -> n_cam f c h w', n_cam=6).contiguous()
        volume_feats = rearrange(volume_feats, '(b f n_cam) c h w -> (b n_cam) f c h w', n_cam=6, f=self.cur_num_frames).contiguous()

        return volume_feats



    @classmethod
    def from_unet_spatio_temporal_condition(
        cls,
        unet: UNetSpatioTemporalConditionModel,
        load_weights_from_unet: bool = True,
        # multivew
        **kwargs,
    ):
        r"""
        Instantiate Multiview unet class from UNet2DConditionModel.

        Parameters:
            unet (`UNet2DConditionModel`):
                UNet model which weights are copied to the ControlNet. Note that all configuration options are also
                copied where applicable.
        """

        unet_spatio_temporal_condition_multiview = cls(
            **unet.config,
            # multivew
            **kwargs,
        )

        unet_state_dict = unet.state_dict()
        # for k in list(unet_state_dict.keys()):
        #     if "time_embed" in k:  # duplicate a new timestep embedding from the pretrained weights
        #         unet_state_dict[k.replace("time_embedding", "cond_time_stack_embedding")] = unet_state_dict[k]

        if load_weights_from_unet:
            # missing_keys, unexpected_keys = unet_spatio_temporal_condition_multiview.load_state_dict(
            #     unet.state_dict(), strict=False)
            missing_keys, unexpected_keys = unet_spatio_temporal_condition_multiview.load_state_dict(
                unet_state_dict, strict=False)
            # logging.info(
            #     f"[UNetSpatioTemporalConditionModelMultiview] load pretrained with "
            #     f"missing_keys: {missing_keys}; "
            #     f"unexpected_keys: {unexpected_keys}")
            logger.info(
                f"[UNetSpatioTemporalConditionModelMultiview] load pretrained with "
                f"missing_keys: {missing_keys}; "
                f"unexpected_keys: {unexpected_keys}")

        return unet_spatio_temporal_condition_multiview

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],

        # cond_mask: Optional[torch.Tensor],

        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,

        occ_latent = None,
        metas = None,

    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.Tensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead
                of a plain tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is
                returned, otherwise a `tuple` is returned where the first element is the sample tensor.
        """


        volume_feats = self.convert_3d_to_2d(occ_latent, metas)
        sample = sample + volume_feats

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        # t_emb = t_emb.repeat_interleave(num_frames, dim=0)
        # cond_mask_ = cond_mask.reshape(-1, 1)
        # emb = self.cond_time_stack_embedding(t_emb) * cond_mask_ + self.time_embedding(t_emb) * (1 - cond_mask_)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)

        # aug_emb = aug_emb.repeat_interleave(num_frames, dim=0)

        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.conv_in(sample)
        # sample = self.custom_conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample,)

        return UNetSpatioTemporalConditionOutput(sample=sample)
