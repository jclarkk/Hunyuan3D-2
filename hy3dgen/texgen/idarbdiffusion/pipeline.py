import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from typing import List

from .data.custom_mv_dataset import CustomMVDataset
from .models.unet_dr2d_condition import UNetDR2DConditionModel
from .pipelines.pipeline_idarbdiffusion import IDArbDiffusionPipeline


class IDArbPipeline:

    @classmethod
    def from_pretrained(cls, device):
        text_encoder = CLIPTextModel.from_pretrained("lizb6626/IDArb", subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained("lizb6626/IDArb", subfolder="tokenizer")
        feature_extractor = CLIPImageProcessor.from_pretrained("lizb6626/IDArb", subfolder="feature_extractor")
        vae = AutoencoderKL.from_pretrained("lizb6626/IDArb", subfolder="vae")
        scheduler = DDIMScheduler.from_pretrained("lizb6626/IDArb", subfolder="scheduler")
        unet = UNetDR2DConditionModel.from_pretrained("lizb6626/IDArb", subfolder="unet")
        pipeline = IDArbDiffusionPipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            vae=vae,
            unet=unet,
            safety_checker=None,
            scheduler=scheduler,
        )
        pipeline.to(device)
        return cls(pipeline, device)

    def __init__(self, pipeline: IDArbDiffusionPipeline, device: str):
        self.pipeline = pipeline
        self.device = device

    def __call__(self, images: List[Image.Image]):
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            self.pipeline.unet.enable_xformers_memory_efficient_attention()
            print(f'Use xformers version: {xformers_version}')

        weight_dtype = torch.float16

        dataset = CustomMVDataset(images, num_views=len(images))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        Nd = 3

        for i, batch in enumerate(dataloader):

            imgs_in, imgs_mask, task_ids = batch['imgs_in'], batch['imgs_mask'], batch['task_ids']
            cam_pose = batch['pose']

            imgs_in = imgs_in.to(weight_dtype).to("cuda")
            cam_pose = cam_pose.to(weight_dtype).to("cuda")

            B, Nv, _, H, W = imgs_in.shape

            imgs_in, imgs_mask, task_ids = imgs_in.flatten(0, 1), imgs_mask.flatten(0, 1), task_ids.flatten(0, 2)

            with torch.autocast(self.device):
                out = self.pipeline(
                    imgs_in,
                    task_ids,
                    num_views=Nv,
                    cam_pose=cam_pose,
                    height=H,
                    width=W,
                    guidance_scale=1.7,  # We need to experiment with this value
                    output_type='pt',
                    num_images_per_prompt=1,
                    eta=0.1,
                ).images

                out = out.view(B, Nv, Nd, *out.shape[1:])
                out = out.detach().cpu().numpy()

            # Extract PBR maps
            pbr_maps = []
            for i_map in range(Nv):
                pbr_maps.append({
                    "albedo": self.extract_albedo(out[0, i_map]),
                    "normal": self.extract_normal(out[0, i_map]),
                    "roughness": self.extract_roughness(out[0, i_map]),
                    "metalness": self.extract_metalness(out[0, i_map])
                })
            return pbr_maps

    def extract_albedo(self, img_nd):
        return self.extract_pbr_channel(img_nd, 0)

    def extract_normal(self, img_nd):
        return self.extract_pbr_channel(img_nd, 1)

    def extract_roughness(self, img_nd):
        return self.extract_pbr_channel(img_nd, 2, channel_idx=1)

    def extract_metalness(self, img_nd):
        return self.extract_pbr_channel(img_nd, 2, channel_idx=0)

    @staticmethod
    def extract_pbr_channel(img_nd, idx, channel_idx=None):
        if idx >= img_nd.shape[0]:
            raise IndexError(f"Tried to access index {idx} in PBR output, but shape is {img_nd.shape}")

        channel = img_nd[idx]

        if channel_idx is not None:
            if channel.shape[0] <= channel_idx:
                raise IndexError(f"Tried to access sub-channel {channel_idx} in shape {channel.shape}")
            channel = channel[channel_idx:channel_idx + 1]

        channel = (channel * 255).astype(np.uint8)

        if channel.shape[0] == 1:
            channel = channel.squeeze(0)

        return np.transpose(channel, (1, 2, 0)) if channel.ndim == 3 else channel

    @staticmethod
    def combine_roughness_metalness(roughness_map: np.ndarray, metalness_map: np.ndarray) -> Image.Image:
        """Pack roughness into G channel and metalness into B channel."""
        metalness = np.array(metalness_map)
        roughness = np.array(roughness_map)

        # Convert grayscale to single channel
        if metalness.ndim == 3:
            metalness = metalness[..., 0]
        if roughness.ndim == 3:
            roughness = roughness[..., 0]

        # Convert to 8-bit range if necessary
        if metalness.max() <= 1:
            metalness = (metalness * 255).astype(np.uint8)
        if roughness.max() <= 1:
            roughness = (roughness * 255).astype(np.uint8)

        # Pack into R=0, G=roughness, B=metalness
        height, width = roughness.shape
        zero_channel = np.zeros((height, width), dtype=np.uint8)
        packed_image = np.stack([zero_channel, roughness, metalness], axis=-1)

        return Image.fromarray(packed_image, mode='RGB')