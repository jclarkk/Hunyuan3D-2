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
import time
from typing import Union, Optional

import numpy as np
import torch
from PIL import Image

from .differentiable_renderer.mesh_render import MeshRender
from .mvadapter.pipeline import MVAdapterPipelineWrapper
from .upscalers.pipelines import AuraSRUpscalerPipeline, InvSRUpscalerPipeline, FluxUpscalerPipeline, \
    RealESRGANUpscalerPipeline
from .utils.dehighlight_utils import Light_Shadow_Remover
from .utils.imagesuper_utils import Image_Super_Net
from .utils.multiview_utils import Multiview_Diffusion_Net
from .utils.uv_warp_utils import mesh_uv_wrap, bpy_unwrap_mesh, open3d_mesh_uv_wrap

logger = logging.getLogger(__name__)


class Hunyuan3DTexGenConfig:

    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path, mv_model='hunyuan3d-paint-v2-0',
                 use_delight=False, device='cuda'):
        self.device = device
        self.mv_model = mv_model
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path
        self.use_delight = use_delight

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.1, 0.1]

        self.candidate_camera_azims_enhanced = [0, 90, 180, 270, 0, 180, 90, 270, 45, 135, 225, 310]
        self.candidate_camera_elevs_enhanced = [0, 0, 0, 0, 90, -90, -45, -45, 15, 15, 15, 15]
        self.candidate_view_weights_enhanced = [1, 0.2, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        self.render_size = 2048
        self.texture_size = 1024
        self.bake_exp = 4
        self.merge_method = 'fast'


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(cls, model_path, mv_model='hunyuan3d-paint-v2-0', use_delight=False):
        original_model_path = model_path
        if not os.path.exists(model_path):
            # try local path
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, model_path))

            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, mv_model)

            if 'hunyuan3d' in mv_model and not os.path.exists(multiview_model_path):
                try:
                    import huggingface_hub
                    # download from huggingface
                    model_path = huggingface_hub.snapshot_download(repo_id=original_model_path)
                    delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
                    multiview_model_path = os.path.join(model_path, 'hunyuan3d-paint-v2-0')
                except ImportError:
                    logger.warning(
                        "You need to install HuggingFace Hub to load models from the hub."
                    )
                    raise RuntimeError(f"Model path {model_path} not found")

            return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path,
                                             mv_model=mv_model, use_delight=use_delight))

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
        if self.config.use_delight:
            self.models['delight_model'] = Light_Shadow_Remover(self.config)
            print('Delight model loaded')
        print(f'Loading multiview model: {self.config.mv_model}')
        if self.config.mv_model == 'hunyuan3d-paint-v2-0' or self.config.mv_model == 'hunyuan3d-paint-v2-0-turbo':
            self.models['multiview_model'] = Multiview_Diffusion_Net(self.config)
        elif self.config.mv_model == 'mv-adapter':
            self.models['multiview_model'] = MVAdapterPipelineWrapper.from_pretrained(device=self.config.device)
        print('Multiview model loaded')

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        if self.models.get('delight_model') is not None:
            self.models['delight_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        self.models['multiview_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            if self.config.mv_model == 'mv-adapter':
                bg_color = [0.5, 0.5, 0.5]
            else:
                bg_color = [1, 1, 1]

            normal_map = self.render.render_normal(
                elev, azim, use_abs_coor=use_abs_coor, return_type='th', bg_color=bg_color
            )
            # Remove batch dimension if present
            if normal_map.dim() == 4 and normal_map.shape[0] == 1:
                normal_map = normal_map.squeeze(0)

            normal_map_pil = Image.fromarray((normal_map.cpu().numpy() * 255).astype(np.uint8))
            normal_maps.append(normal_map_pil)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            if self.config.mv_model == 'mv-adapter':
                bg_color = [0.5, 0.5, 0.5]
            else:
                bg_color = [1, 1, 1]

            position_map = self.render.render_position(
                elev, azim, return_type='th', bg_color=bg_color
            )
            # Remove batch dimension if present
            if position_map.dim() == 4 and position_map.shape[0] == 1:
                position_map = position_map.squeeze(0)

            position_map_pil = Image.fromarray((position_map.cpu().numpy() * 255).astype(np.uint8))
            position_maps.append(position_map_pil)

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
            if self.config.use_delight:
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
    def __call__(self,
                 mesh,
                 image,
                 unwrap_method='xatlas',
                 upscale_model=None,
                 enhance_texture_angles=False,
                 pbr=False,
                 debug=False,
                 texture_size=1024,
                 seed=42):
        self.config.texture_size = texture_size
        self.render.set_default_texture_resolution(texture_size)

        if isinstance(image, str):
            image_prompt = Image.open(image)
        else:
            image_prompt = image

        image_prompt = self.recenter_image(image_prompt)

        if self.config.use_delight:
            print('Removing light and shadow...')
            t0 = time.time()
            height, width = 512, 512
            if self.config.mv_model == 'mv-adapter':
                height = 768
                width = 768

            image_prompt = self.models['delight_model'](image_prompt, height, width)
            t1 = time.time()
            print(f"Light and shadow removal took {t1 - t0:.2f} seconds")

        print('Wrapping UV...')
        t0 = time.time()
        if unwrap_method == 'open3d':
            mesh = open3d_mesh_uv_wrap(mesh, resolution=texture_size)
        elif unwrap_method == 'bpy':
            mesh = bpy_unwrap_mesh(mesh)
        elif unwrap_method == 'xatlas':
            mesh = mesh_uv_wrap(mesh)
        else:
            raise ValueError(f"Invalid unwrap method {unwrap_method}")
        t1 = time.time()
        print(f"UV wrapping took {t1 - t0:.2f} seconds")

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
        t0 = time.time()
        normal_maps = self.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
        position_maps = self.render_position_multiview(
            selected_camera_elevs, selected_camera_azims)
        t1 = time.time()
        print(f"Rendering normal and position maps took {t1 - t0:.2f} seconds")
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
        t0 = time.time()
        if self.config.mv_model == 'hunyuan3d-paint-v2-0':
            multiviews = self.models['multiview_model'](image_prompt, normal_maps + position_maps, camera_info)
        elif self.config.mv_model == 'mv-adapter':
            multiviews = self.models['multiview_model'](mesh, image_prompt, normal_maps, position_maps, camera_info,
                                                        len(selected_camera_azims), seed=seed)
        else:
            raise ValueError(f"Invalid MV model {self.config.mv_model}")
        t1 = time.time()
        print(f"Generating multiviews took {t1 - t0:.2f} seconds")

        if debug:
            for i in range(len(multiviews)):
                image = multiviews[i]
                image.save(f'debug_multiview_{i}.png')

        if upscale_model == 'Aura':
            upscaler = AuraSRUpscalerPipeline.from_pretrained()
        elif upscale_model == 'RealESRGAN':
            upscaler = RealESRGANUpscalerPipeline.from_pretrained(self.config.device)
        elif upscale_model == 'InvSR':
            upscaler = InvSRUpscalerPipeline.from_pretrained(self.config.device)
        elif upscale_model == 'Flux':
            upscaler = FluxUpscalerPipeline.from_pretrained(self.config.device)
        elif upscale_model == 'SD-Upscaler':
            upscaler = Image_Super_Net(self.config.device)
        else:
            upscaler = None

        if upscaler is not None:
            print('Upscaler model loaded')
            t0 = time.time()

            new_multiviews = []

            from tqdm import tqdm
            for i in tqdm(range(len(multiviews)), desc="Upscaling multiviews"):
                rgb_img = multiviews[i].convert("RGB")

                if i < 6:
                    rgb_img = upscaler(rgb_img)

                    if debug:
                        rgb_img.save(f'debug_multiview_{i}_upscaled.png')

                rgb_img = rgb_img.resize((self.config.texture_size, self.config.texture_size))

                new_multiviews.append(rgb_img)

            del upscaler
            torch.cuda.empty_cache()

            multiviews = new_multiviews

            t1 = time.time()
            print(f"Upscaling multiviews took {t1 - t0:.2f} seconds")
        else:
            for i in range(len(multiviews)):
                multiviews[i] = multiviews[i].resize(
                    (self.config.texture_size, self.config.texture_size))
                if debug:
                    multiviews[i].save(f'debug_multiview_{i}.png')

        print('Baking texture...')
        t0 = time.time()
        texture, mask = self.bake_from_multiview(multiviews,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=self.config.merge_method)
        t1 = time.time()
        print(f"Texture baking took {t1 - t0:.2f} seconds")

        normal_texture, metallic_roughness_texture, metallic_factor, roughness_factor = None, None, None, None
        if pbr:
            t0 = time.time()
            from .pbr.pipelines import RGB2XPipeline
            pbr_pipeline = RGB2XPipeline.from_pretrained("cuda")

            # Do it in batches of 6
            pre_pbr_multiviews = [view.resize((1024, 1024)) for view in multiviews[:6]]
            albedo_multiviews, metallic_multiviews, normal_multiviews, roughness_multiviews = (
                self.generate_pbr_for_batch(pbr_pipeline, pre_pbr_multiviews))

            for i in range(len(albedo_multiviews)):
                albedo_multiviews[i] = albedo_multiviews[i].resize((self.config.texture_size, self.config.texture_size))
                normal_multiviews[i] = normal_multiviews[i].resize((self.config.texture_size, self.config.texture_size))
                roughness_multiviews[i] = roughness_multiviews[i].resize(
                    (self.config.texture_size, self.config.texture_size))
                metallic_multiviews[i] = metallic_multiviews[i].resize(
                    (self.config.texture_size, self.config.texture_size))

                if debug:
                    albedo_multiviews[i].save(f'debug_albedo_multiview_{i}.png')
                    normal_multiviews[i].save(f'debug_normal_multiview_{i}.png')
                    roughness_multiviews[i].save(f'debug_roughness_multiview_{i}.png')
                    metallic_multiviews[i].save(f'debug_metallic_multiview_{i}.png')

            normal_texture = None

            print('Baking roughness PBR texture...')
            roughness_texture, _ = self.bake_from_multiview(roughness_multiviews,
                                                            self.config.candidate_camera_elevs,
                                                            self.config.candidate_camera_azims,
                                                            self.config.candidate_view_weights,
                                                            method=self.config.merge_method)
            roughness_texture = roughness_texture.cpu().numpy()
            print('Baking metallic PBR texture...')
            metallic_texture, _ = self.bake_from_multiview(metallic_multiviews,
                                                           self.config.candidate_camera_elevs,
                                                           self.config.candidate_camera_azims,
                                                           self.config.candidate_view_weights,
                                                           method=self.config.merge_method)
            metallic_texture = metallic_texture.cpu().numpy()
            metallic_roughness_texture = pbr_pipeline.combine_roughness_metalness(
                roughness_texture,
                metallic_texture
            )

            roughness_factor, metallic_factor = 0.8, 0.8

            t1 = time.time()
            print(f"PBR texture baking took {t1 - t0:.2f} seconds")

        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        print('Inpainting texture...')
        t0 = time.time()
        texture = self.texture_inpaint(texture, mask_np)
        t1 = time.time()
        print(f"Texture inpainting took {t1 - t0:.2f} seconds")

        self.render.set_texture(texture)
        textured_mesh = self.render.save_mesh(
            normal_texture,
            metallic_roughness_texture,
            metallic_factor,
            roughness_factor
        )

        return textured_mesh

    def generate_pbr_for_batch(self, pbr_pipeline, pre_pbr_multiviews):
        pre_pbr_image = self.concatenate_images(pre_pbr_multiviews)
        print('Generating PBR textures...')
        pbr_dict = pbr_pipeline(pre_pbr_image)
        albedo = pbr_dict['albedo']
        normal = pbr_dict['normal']
        roughness = pbr_dict['roughness']
        metallic = pbr_dict['metallic']
        albedo_multiviews = self.split_images(albedo)
        normal_multiviews = self.split_images(normal)
        roughness_multiviews = self.split_images(roughness)
        metallic_multiviews = self.split_images(metallic)
        return albedo_multiviews, metallic_multiviews, normal_multiviews, roughness_multiviews

    @staticmethod
    def concatenate_images(image_list):
        grid_size = (3, 2)
        output_size = (1024 * grid_size[0], 1024 * grid_size[1])

        big_image = Image.new("RGB", output_size)

        for idx, img in enumerate(image_list):
            x_offset = (idx % grid_size[0]) * 1024
            y_offset = (idx // grid_size[0]) * 1024
            big_image.paste(img, (x_offset, y_offset))

        return big_image

    @staticmethod
    def split_images(big_image):
        grid_size = (3, 2)
        image_list = []

        for row in range(grid_size[1]):
            for col in range(grid_size[0]):
                x_offset = col * 1024
                y_offset = row * 1024
                cropped = big_image.crop((x_offset, y_offset, x_offset + 1024, y_offset + 1024))
                image_list.append(cropped)

        return image_list
