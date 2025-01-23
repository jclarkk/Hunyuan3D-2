# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.
import logging
import numpy as np
import os
import torch
from PIL import Image
from aura_sr import AuraSR
from typing import List

from .differentiable_renderer.mesh_render import MeshRender
from .utils.dehighlight_utils import Light_Shadow_Remover
from .utils.multiview_utils import Multiview_Diffusion_Net
from .utils.uv_warp_utils import mesh_uv_wrap

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.
# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.
# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

logger = logging.getLogger(__name__)


class Hunyuan3DTexGenConfig:

    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path):
        self.device = 'cuda'
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        self.render_size = 2048
        self.texture_size = 1024
        self.bake_exp = 4
        self.merge_method = 'fast'


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(cls, model_path):
        original_model_path = model_path
        if not os.path.exists(model_path):
            # try local path
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, model_path))

            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, 'hunyuan3d-paint-v2-0')

            if not os.path.exists(delight_model_path) or not os.path.exists(multiview_model_path):
                try:
                    import huggingface_hub
                    # download from huggingface
                    model_path = huggingface_hub.snapshot_download(repo_id=original_model_path)
                    delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
                    multiview_model_path = os.path.join(model_path, 'hunyuan3d-paint-v2-0')
                    return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path))
                except ImportError:
                    logger.warning(
                        "You need to install HuggingFace Hub to load models from the hub."
                    )
                    raise RuntimeError(f"Model path {model_path} not found")
            else:
                return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path))

        raise FileNotFoundError(f"Model path {original_model_path} not found and we could not find it at huggingface")

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size)

        self.load_models()

    def load_models(self):
        # empty cude cache
        torch.cuda.empty_cache()
        # Load model
        self.models['delight_model'] = Light_Shadow_Remover(self.config)
        self.models['multiview_model'] = Multiview_Diffusion_Net(self.config)

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render.render_normal(
                elev, azim, use_abs_coor=use_abs_coor, return_type='pl')
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(
                elev, azim, return_type='pl')
            position_maps.append(position_map)

        return position_maps

    def bake_from_multiview(self, views, camera_elevs,
                            camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        for view, camera_elev, camera_azim, weight in zip(
                views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim)
            project_cos_map = weight * (project_cos_map ** self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)

        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)

            final_texture_linear_torch = torch.tensor(texture).float()
            texture = self.render.color_rgb_to_srgb(final_texture_linear_torch)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8

    def texture_inpaint(self, texture, mask):

        texture_np = self.render.uv_inpaint(texture, mask)
        texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture

    @torch.no_grad()
    def __call__(self, mesh, image, texture_size=2048, upscale=False, pbr=False):
        self.render.set_default_texture_resolution(texture_size)

        if isinstance(image, str):
            image_prompt = Image.open(image)
        else:
            image_prompt = image

        image_prompt = self.models['delight_model'](image_prompt)

        mesh = mesh_uv_wrap(mesh)

        self.render.load_mesh(mesh)

        selected_camera_elevs, selected_camera_azims, selected_view_weights = \
            self.config.candidate_camera_elevs, self.config.candidate_camera_azims, self.config.candidate_view_weights

        normal_maps = self.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
        position_maps = self.render_position_multiview(
            selected_camera_elevs, selected_camera_azims)

        camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[
            elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                       zip(selected_camera_azims, selected_camera_elevs)]
        multiviews = self.models['multiview_model'](image_prompt, normal_maps + position_maps, camera_info)

        if upscale:
            if texture_size == 4096:
                # Resize multiviews to 1024x1024 first
                for i in range(len(multiviews)):
                    multiviews[i] = multiviews[i].resize((1024, 1024))

            # Multi view images are 512x512 so we will use Real-ESRGAN to upscale them to 2048x2048
            new_multiviews = []

            upscaler = load_upscaler()
            for i in range(len(multiviews)):
                rgb_img = multiviews[i].convert("RGB")
                upscaled_img = upscaler.upscale_4x(rgb_img)
                new_multiviews.append(upscaled_img)

            multiviews = new_multiviews
        else:
            for i in range(len(multiviews)):
                multiviews[i] = multiviews[i].resize(
                    (self.config.render_size, self.config.render_size))

        texture, mask = self.bake_from_multiview(multiviews,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=self.config.merge_method)

        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        normal_map = None
        metalness_roughness_map = None

        if pbr:
            pbr_pipeline = IDArbPipeline()
            pbr_images = pbr_pipeline.run(multiviews)

            normal_maps = []
            roughness_maps = []
            metalness_maps = []

            for pbr_image_maps in pbr_images:
                normal_maps.append(pbr_image_maps['normal'])
                roughness_maps.append(pbr_image_maps['roughness'])
                metalness_maps.append(pbr_image_maps['metalness'])

            # Bake PBR into UV space
            normal_map, normal_mask = self.bake_from_multiview(
                normal_maps,
                selected_camera_elevs,
                selected_camera_azims,
                selected_view_weights,
                method=self.config.merge_method
            )

            # 3. Generate roughness_map_views
            roughness_map, roughness_mask = self.bake_from_multiview(
                roughness_maps,
                selected_camera_elevs,
                selected_camera_azims,
                selected_view_weights,
                method=self.config.merge_method
            )

            # 4. Generate metalness_map_views
            metalness_map, metalness_mask = self.bake_from_multiview(
                metalness_maps,
                selected_camera_elevs,
                selected_camera_azims,
                selected_view_weights,
                method=self.config.merge_method
            )

            metalness_roughness_map = self.combine_roughness_metalness(np.asarray(roughness_map), np.asarray(metalness_map))

        texture = self.texture_inpaint(texture, mask_np)

        self.render.set_texture(texture)
        textured_mesh = self.render.save_mesh(normal_map, metalness_roughness_map)

        return textured_mesh

    @staticmethod
    def combine_roughness_metalness(roughness_map: np.ndarray, metalness_map: np.ndarray) -> Image.Image:
        """
        roughness_map: HxW or HxWx1, values in [0..1]
        metalness_map: HxW or HxWx1, values in [0..1]

        Returns: PIL Image (RGB),
                 R = roughness, G = metalness, B = 0
        """

        # Ensure both are HxW and float [0..1]
        if roughness_map.ndim == 3 and roughness_map.shape[2] == 3:
            # Possibly the roughness is already 3-channel? Just pick one.
            roughness_map = roughness_map[..., 0]
        if metalness_map.ndim == 3 and metalness_map.shape[2] == 3:
            metalness_map = metalness_map[..., 0]

        # Convert from [0..1] float -> [0..255] uint8
        roughness_8 = (roughness_map * 255).astype(np.uint8)
        metalness_8 = (metalness_map * 255).astype(np.uint8)

        # Make sure dimensions match
        if roughness_8.shape != metalness_8.shape:
            raise ValueError("Roughness and metalness maps must have the same shape.")

        # Stack into a 3-channel array: (H, W, 3)
        # R=roughness, G=metalness, B=0  (you could also do B=255 or a third channel if needed)
        height, width = roughness_8.shape
        blue_channel = np.zeros((height, width), dtype=np.uint8)  # or np.full((height,width), 255, dtype=np.uint8)

        combined_map = np.stack([roughness_8, metalness_8, blue_channel], axis=-1)

        # Convert to a PIL image in RGB mode
        combined_pil = Image.fromarray(combined_map, mode='RGB')
        return combined_pil


def load_upscaler():
    upscaler = AuraSR.from_pretrained()
    return upscaler


class IDArbPipeline:

    @staticmethod
    def load_pipeline():
        from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
        from diffusers import AutoencoderKL, DDIMScheduler
        from idarbdiffusion.models.unet_dr2d_condition import UNetDR2DConditionModel
        from idarbdiffusion.pipelines.pipeline_idarbdiffusion import IDArbDiffusionPipeline

        """
        load pipeline from hub
        or load from local ckpts: pipeline = IDArbDiffusionPipeline.from_pretrained("./pipeckpts")
        """
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
        return pipeline

    def run(self, images: List[Image.Image]):
        pipeline = self.load_pipeline()

        from idarbdiffusion.data.custom_mv_dataset import CustomMVDataset
        from diffusers.utils.import_utils import is_xformers_available
        from packaging import version

        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            pipeline.unet.enable_xformers_memory_efficient_attention()
            print(f'Use xformers version: {xformers_version}')

        weight_dtype = torch.float16

        dataset = CustomMVDataset(images, num_views=len(images))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        Nd = 3

        pbr_images = []
        for i, batch in enumerate(dataloader):

            imgs_in, imgs_mask, task_ids = batch['imgs_in'], batch['imgs_mask'], batch['task_ids']
            cam_pose = batch['pose']
            imgs_name = batch['data_name']

            imgs_in = imgs_in.to(weight_dtype).to("cuda")
            cam_pose = cam_pose.to(weight_dtype).to("cuda")

            B, Nv, _, H, W = imgs_in.shape

            imgs_in, imgs_mask, task_ids = imgs_in.flatten(0, 1), imgs_mask.flatten(0, 1), task_ids.flatten(0, 2)

            with torch.autocast("cuda"):
                out = pipeline(
                    imgs_in,
                    task_ids,
                    num_views=Nv,
                    cam_pose=cam_pose,
                    height=H, width=W,
                    # generator=generator,
                    guidance_scale=1.0,
                    output_type='pt',
                    num_images_per_prompt=1,
                    eta=1.0,
                ).images

                out = out.view(B, Nv, Nd, *out.shape[1:])
                out = out.detach().cpu().numpy()

                current_image_maps = {}
                for i in range(B):
                    current_image_maps[imgs_name[i]] = out[i]
                pbr_images.append(current_image_maps)

        return pbr_images
