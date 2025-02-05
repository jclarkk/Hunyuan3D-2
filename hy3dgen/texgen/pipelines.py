# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.
import sys

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


import logging
import os

import numpy as np
import torch
from PIL import Image

from .differentiable_renderer.mesh_render import MeshRender
from .upscalers.pipelines import AuraSRUpscalerPipeline, InvSRUpscalerPipeline, FluxUpscalerPipeline
from .utils.dehighlight_utils import Light_Shadow_Remover
from .utils.multiview_utils import Multiview_Diffusion_Net
from .utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


class Hunyuan3DTexGenConfig:

    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path):
        self.device = 'cuda'
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.1, 0.1]

        self.candidate_camera_azims_enhanced = [0, 90, 180, 270, 0, 180, 90, 270, 45, 135, 225, 310, 45, 135, 225, 310]
        self.candidate_camera_elevs_enhanced = [0, 0, 0, 0, 90, -90, -45, -45, 15, 15, 15, 15, -15, -15, -15, -15]
        self.candidate_view_weights_enhanced = [1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                                0.1]

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
        # empty cuda cache
        torch.cuda.empty_cache()
        # Load model
        self.models['delight_model'] = Light_Shadow_Remover(self.config)
        print('Delight model loaded')
        self.models['multiview_model'] = Multiview_Diffusion_Net(self.config)
        print('Multiview model loaded')

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

            texture = texture.clone().detach().float()
            texture = self.render.color_rgb_to_srgb(texture)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8

    def texture_inpaint(self, texture, mask):

        texture_np = self.render.uv_inpaint(texture, mask)
        texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture

    def recenter_image(self, image, border_ratio=0.2):
        if image.mode == 'RGB':
            return image
        elif image.mode == 'L':
            image = image.convert('RGB')
            return image

        alpha_channel = np.array(image)[:, :, 3]
        non_zero_indices = np.argwhere(alpha_channel > 0)
        if non_zero_indices.size == 0:
            raise ValueError("Image is fully transparent")

        min_row, min_col = non_zero_indices.min(axis=0)
        max_row, max_col = non_zero_indices.max(axis=0)

        cropped_image = image.crop((min_col, min_row, max_col + 1, max_row + 1))

        width, height = cropped_image.size
        border_width = int(width * border_ratio)
        border_height = int(height * border_ratio)

        new_width = width + 2 * border_width
        new_height = height + 2 * border_height

        square_size = max(new_width, new_height)

        new_image = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))

        paste_x = (square_size - new_width) // 2 + border_width
        paste_y = (square_size - new_height) // 2 + border_height

        new_image.paste(cropped_image, (paste_x, paste_y))
        return new_image

    @torch.no_grad()
    def __call__(self, mesh, image, upscale_model='Aura', enhance_texture_angles=False, pbr=False, debug=False):

        if isinstance(image, str):
            image_prompt = Image.open(image)
        else:
            image_prompt = image

        image_prompt = self.recenter_image(image_prompt)

        print('Removing light and shadow...')
        image_prompt = self.models['delight_model'](image_prompt)

        del self.models['delight_model']

        print('Wrapping UV...')
        mesh = mesh_uv_wrap(mesh)

        self.render.load_mesh(mesh)

        if enhance_texture_angles:
            selected_camera_elevs, selected_camera_azims, selected_view_weights = \
                (self.config.candidate_camera_elevs_enhanced, self.config.candidate_camera_azims_enhanced,
                 self.config.candidate_view_weights_enhanced)
        else:
            selected_camera_elevs, selected_camera_azims, selected_view_weights = \
                (self.config.candidate_camera_elevs, self.config.candidate_camera_azims,
                 self.config.candidate_view_weights)

        print('Rendering normal maps...')
        normal_maps = self.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
        position_maps = self.render_position_multiview(
            selected_camera_elevs, selected_camera_azims)
        if debug:
            for i in range(len(normal_maps)):
                normal_maps[i].save(f'debug_normal_map_{i}.png')
                position_maps[i].save(f'debug_position_map_{i}.png')

        if enhance_texture_angles:
            camera_info = [
                (((azim // 30) + 9) % 12) // {
                    -90: 3, -45: 3, -20: 1, -15: 1, 0: 1, 15: 1, 20: 1, 90: 3
                }[elev] + {
                    -90: 36, -45: 36, -20: 0, -15: 0, 0: 12, 15: 24, 20: 24, 90: 40
                }[elev]
                for azim, elev in
                zip(self.config.candidate_camera_azims_enhanced, self.config.candidate_camera_elevs_enhanced)
            ]
        else:
            camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[
                elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                           zip(selected_camera_azims, selected_camera_elevs)]

        print('Generate multiviews...')
        multiviews = self.models['multiview_model'](image_prompt, normal_maps + position_maps, camera_info)

        del self.models['multiview_model']
        torch.cuda.empty_cache()

        if upscale_model == 'Aura':
            upscaler = AuraSRUpscalerPipeline.from_pretrained()
        elif upscale_model == 'InvSR':
            upscaler = InvSRUpscalerPipeline.from_pretrained(self.config.device)
        elif upscale_model == 'Flux':
            upscaler = FluxUpscalerPipeline.from_pretrained(self.config.device)
        else:
            raise ValueError(f"Unknown upscale model {upscale_model}")

        if upscale_model is not None:
            print('Upscaler model loaded')

            new_multiviews = []

            from tqdm import tqdm
            for i in tqdm(range(len(multiviews)), desc="Upscaling multiviews"):
                rgb_img = multiviews[i].convert("RGB")
                if debug:
                    rgb_img.save(f'debug_multiview_{i}.png')

                if i < 6:
                    rgb_img = upscaler(rgb_img)

                    if debug:
                        rgb_img.save(f'debug_multiview_{i}_upscaled.png')

                rgb_img = rgb_img.resize((self.config.texture_size, self.config.texture_size))

                new_multiviews.append(rgb_img)

            del upscaler
            torch.cuda.empty_cache()

            multiviews = new_multiviews

        else:
            for i in range(len(multiviews)):
                multiviews[i] = multiviews[i].resize(
                    (self.config.render_size, self.config.render_size))

        print('Baking texture...')
        texture, mask = self.bake_from_multiview(multiviews,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=self.config.merge_method)

        normal_texture, metallic_roughness_texture, metallic_factor, roughness_factor = None, None, None, None
        if pbr:
            from .pbr.pipelines import RGB2XPipeline
            pbr_pipeline = RGB2XPipeline.from_pretrained(self.config.device)

            pbr_camera_elevs = [angle for angle in selected_camera_elevs[:6]]
            pbr_camera_azims = [angle for angle in selected_camera_azims[:6]]
            pbr_camera_weights = [weight for weight in selected_view_weights[:6]]

            # Downscale multiviews to 512x512 - use only 6 views
            pre_pbr_multiviews = [view.resize((512, 512)) for view in multiviews[:6]]
            pre_pbr_image = self.concatenate_images(pre_pbr_multiviews)

            pbr_dict = pbr_pipeline(pre_pbr_image)
            albedo = pbr_dict['albedo']
            normal = pbr_dict['normal']
            roughness = pbr_dict['roughness']
            metallic = pbr_dict['metallic']

            albedo_multiviews = self.split_images(albedo)
            normal_multiviews = self.split_images(normal)
            roughness_multiviews = self.split_images(roughness)
            metallic_multiviews = self.split_images(metallic)

            for i in range(len(albedo_multiviews)):
                albedo_multiviews[i] = albedo_multiviews[i].resize((self.config.texture_size, self.config.texture_size))
                normal_multiviews[i] = normal_multiviews[i].resize((self.config.texture_size, self.config.texture_size))
                roughness_multiviews[i] = roughness_multiviews[i].resize((self.config.texture_size, self.config.texture_size))
                metallic_multiviews[i] = metallic_multiviews[i].resize((self.config.texture_size, self.config.texture_size))

            print('Baking albedo PBR texture...')
            albedo_texture, mask = self.bake_from_multiview(albedo_multiviews,
                                                            pbr_camera_elevs,
                                                            pbr_camera_azims,
                                                            pbr_camera_weights,
                                                            method=self.config.merge_method)
            print('Baking normal PBR texture...')
            normal_texture, _ = self.bake_from_multiview(normal_multiviews,
                                                         pbr_camera_elevs,
                                                         pbr_camera_azims,
                                                         pbr_camera_weights,
                                                         method=self.config.merge_method)
            normal_texture = normal_texture.squeeze(0)
            normal_texture_np = normal_texture.cpu().numpy()
            if normal_texture_np.dtype == np.float32:
                normal_texture_np = (normal_texture_np * 255).clip(0, 255).astype(np.uint8)
            normal_texture = Image.fromarray(normal_texture_np)
            print('Baking roughness PBR texture...')
            roughness_texture, _ = self.bake_from_multiview(roughness_multiviews,
                                                            pbr_camera_elevs,
                                                            pbr_camera_azims,
                                                            pbr_camera_weights,
                                                            method=self.config.merge_method)
            roughness_texture = roughness_texture.cpu().numpy()
            print('Baking metallic PBR texture...')
            metallic_texture, _ = self.bake_from_multiview(metallic_multiviews,
                                                           pbr_camera_elevs,
                                                           pbr_camera_azims,
                                                           pbr_camera_weights,
                                                           method=self.config.merge_method)
            metallic_texture = metallic_texture.cpu().numpy()
            metallic_roughness_texture = pbr_pipeline.combine_roughness_metalness(
                roughness_texture,
                metallic_texture
            )
            metallic_factor, roughness_factor = pbr_pipeline.analyze_texture(texture)

            texture = albedo_texture

        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        print('Inpainting texture...')
        texture = self.texture_inpaint(texture, mask_np)

        self.render.set_texture(texture)
        textured_mesh = self.render.save_mesh(
            normal_texture,
            metallic_roughness_texture,
            metallic_factor,
            roughness_factor
        )

        return textured_mesh

    @staticmethod
    def concatenate_images(image_list):
        """Concatenates 6 images (512x512) into a single 3072x3072 image."""
        grid_size = (3, 2)
        output_size = (512 * grid_size[0], 512 * grid_size[1])

        big_image = Image.new("RGB", output_size)

        for idx, img in enumerate(image_list):
            x_offset = (idx % grid_size[0]) * 512
            y_offset = (idx // grid_size[0]) * 512
            big_image.paste(img, (x_offset, y_offset))

        return big_image

    @staticmethod
    def split_images(big_image):
        """Splits a 3072x3072 image back into 6 images of 512x512."""
        grid_size = (3, 2)
        image_list = []

        for row in range(grid_size[1]):
            for col in range(grid_size[0]):
                x_offset = col * 512
                y_offset = row * 512
                cropped = big_image.crop((x_offset, y_offset, x_offset + 512, y_offset + 512))
                image_list.append(cropped)

        return image_list
