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
from typing import Union, Optional, List

import numpy as np
import torch
from PIL import Image

from .differentiable_renderer.mesh_render import MeshRender
from .mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from .mvadapter.pipelines.pipeline_texture import ModProcessConfig, TexturePipeline
from ..text2image import HunyuanDiTPipeline

logger = logging.getLogger(__name__)


class Hunyuan3DTexGenConfig:

    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path, mv_model='hunyuan3d-paint-v2-0',
                 use_delight=False, device='cpu', mv_adapter_model_class=MVAdapterI2MVSDXLPipeline,
                 baking_pipeline='hunyuan'):
        self.device = device
        self.mv_model = mv_model
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path
        self.use_delight = use_delight

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.1, 0.1]

        self.candidate_camera_azims_enhanced = [
            0, 90, 180, 270, 0, 180,
            90, 270, 45, 135, 225, 310,
            0, 90, 180, 270, 45, 225
        ]
        self.candidate_camera_elevs_enhanced = [
            0, 0, 0, 0, 90, -90,
            -45, -45, 15, 15, 15, 15,
            20, 20, 20, 20, -20, -20
        ]
        self.candidate_view_weights_enhanced = [
            1, 0.4, 0.5, 0.4, 0.1, 0.1,
            0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
            0.2, 0.2, 0.2, 0.2, 0.2, 0.2
        ]

        self.render_size = 2048
        self.texture_size = 1024
        self.bake_exp = 8
        self.merge_method = 'fast'

        self.mv_adapter_model_class = mv_adapter_model_class
        self.mv_adapter_inpaint_weights = "./weights/big-lama.pt"

        self.pipe_dict = {'hunyuan3d-paint-v2-0': 'hunyuanpaint',
                          'hunyuan3d-paint-v2-0-turbo': 'hunyuanpaint-turbo',
                          'mv-adapter': 'mv-adapter'}
        self.pipe_name = self.pipe_dict[mv_model]

        self.baking_pipeline = baking_pipeline


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(cls, model_path, mv_model='hunyuan3d-paint-v2-0', use_delight=False, local_files_only=False,
                        device='cuda', mv_adapter_model_class=MVAdapterI2MVSDXLPipeline, baking_pipeline='hunyuan'):
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

            return cls(Hunyuan3DTexGenConfig(delight_model_path,
                                             multiview_model_path,
                                             mv_model=mv_model,
                                             use_delight=use_delight,
                                             device=device,
                                             mv_adapter_model_class=mv_adapter_model_class,
                                             baking_pipeline=baking_pipeline
                                             ), local_files_only=local_files_only)

        raise FileNotFoundError(f"Model path {original_model_path} not found and we could not find it at huggingface")

    def __init__(self, config, local_files_only=False):
        self.config = config
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size)

        self.load_models(local_files_only=local_files_only)

    def load_models(self, local_files_only=False):
        # empty cuda cache
        torch.cuda.empty_cache()
        # Load model
        if self.config.use_delight:
            from .utils.dehighlight_utils import Light_Shadow_Remover
            self.models['delight_model'] = Light_Shadow_Remover(self.config)
            print('Delight model loaded')
        print(f'Loading multiview model: {self.config.mv_model}')
        self.models['t2i_model'] = None
        if self.config.mv_model == 'hunyuan3d-paint-v2-0' or self.config.mv_model == 'hunyuan3d-paint-v2-0-turbo':
            from .utils.multiview_utils import Multiview_Diffusion_Net
            self.models['multiview_model'] = Multiview_Diffusion_Net(self.config, local_files_only=local_files_only)
            t2i_pipeline = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
                                              local_files_only=local_files_only,
                                              device=self.config.device)
            self.models['t2i_model'] = t2i_pipeline

        elif self.config.mv_model == 'mv-adapter':
            from .mvadapter.pipeline import MVAdapterPipelineWrapper
            self.models['multiview_model'] = MVAdapterPipelineWrapper.from_pretrained(device=self.config.device,
                                                                                      local_files_only=local_files_only,
                                                                                      model_cls=self.config.mv_adapter_model_class)
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
                            camera_azims, view_weights):
        """
        • if <=6 views → original single-pass bake
        • if  >6 views → two-pass: first 6 as primary, rest fill gaps only
        """
        proj_tex, proj_cos = [], []
        for view, elev, azim, w in zip(views, camera_elevs, camera_azims, view_weights):
            tex, cos, _ = self.render.back_project(view, elev, azim)
            proj_tex.append(tex)
            proj_cos.append(w * (cos ** self.config.bake_exp))

        if len(views) <= 6:
            texture, trust = self.render.fast_bake_texture(proj_tex,
                                                           proj_cos)
        else:
            texture, trust = self.render.fast_bake_texture_gap(
                proj_tex[:6], proj_cos[:6],  # primary
                proj_tex[6:], proj_cos[6:]  # secondary
            )

        return texture, trust

    def texture_inpaint(self, texture, mask):

        texture_np = self.render.uv_inpaint(texture, mask)
        texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture

    def recenter_image(self, image, border_ratio=0.2):
        if 'A' not in image.getbands():
            return image.convert('RGB') if image.mode != 'RGB' else image

        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        alpha = np.asarray(image)[..., 3]
        non_zero = np.argwhere(alpha > 0)
        if non_zero.size == 0:
            return image

        min_row, min_col = non_zero.min(axis=0)
        max_row, max_col = non_zero.max(axis=0)

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
                 images=None,
                 prompt=None,
                 unwrap_method='xatlas',
                 upscale_model=None,
                 enhance_texture_angles=False,
                 pbr=False,
                 debug=False,
                 texture_size=1024,
                 seed=42,
                 output_dir='./output',
                 output_name='textured_mesh'):
        if debug:
            os.makedirs('./debug', exist_ok=True)

        self.config.texture_size = texture_size
        self.render.set_default_texture_resolution(texture_size)

        if images is not None and not isinstance(images, List):
            images = [images]

        images_prompt = None
        if images is not None:
            images_prompt = []
            for i in range(len(images)):
                if isinstance(images[i], str):
                    image_prompt = Image.open(images[i])
                else:
                    image_prompt = images[i]
                images_prompt.append(image_prompt)

            images_prompt = [self.recenter_image(image_prompt) for image_prompt in images_prompt]

        if images_prompt is not None and self.config.use_delight:
            print('Removing light and shadow...')
            t0 = time.time()
            images_prompt = [self.models['delight_model'](image_prompt) for image_prompt in images_prompt]

            t1 = time.time()
            print(f"Light and shadow removal took {t1 - t0:.2f} seconds")

        self.render.load_mesh(mesh)

        if enhance_texture_angles:
            selected_camera_elevs, selected_camera_azims, selected_view_weights = \
                (self.config.candidate_camera_elevs_enhanced, self.config.candidate_camera_azims_enhanced,
                 self.config.candidate_view_weights_enhanced)
        else:
            selected_camera_elevs, selected_camera_azims, selected_view_weights = \
                (self.config.candidate_camera_elevs, self.config.candidate_camera_azims,
                 self.config.candidate_view_weights)

        normal_maps, position_maps = None, None
        if self.config.baking_pipeline == 'hunyuan':
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
        if self.config.mv_model in ['hunyuan3d-paint-v2-0', 'hunyuan3d-paint-v2-0-turbo']:

            multiviews = self.models['multiview_model'](images_prompt, normal_maps + position_maps, camera_info)
        elif self.config.mv_model == 'mv-adapter':
            if images_prompt is not None:
                if self.config.baking_pipeline == 'hunyuan':
                    multiviews = self.models['multiview_model'](mesh,
                                                                images_prompt[0],
                                                                normal_maps=normal_maps,
                                                                position_maps=position_maps,
                                                                camera_elevation_deg=selected_camera_elevs,
                                                                camera_azimuth_deg=selected_camera_azims,
                                                                num_views=len(selected_camera_azims),
                                                                seed=seed,
                                                                use_mesh_renderer=False,
                                                                save_debug_images=debug)
                else:
                    multiviews = self.models['multiview_model'](mesh,
                                                                images_prompt[0],
                                                                camera_elevation_deg=selected_camera_elevs,
                                                                camera_azimuth_deg=selected_camera_azims,
                                                                num_views=len(selected_camera_azims),
                                                                seed=seed,
                                                                use_mesh_renderer=True,
                                                                save_debug_images=debug)

            else:
                multiviews = self.models['multiview_model'](mesh,
                                                            normal_maps,
                                                            position_maps,
                                                            prompt=prompt,
                                                            camera_elevation_deg=selected_camera_elevs,
                                                            camera_azimuth_deg=selected_camera_azims,
                                                            num_views=len(selected_camera_azims),
                                                            seed=seed,
                                                            save_debug_images=debug)
        else:
            raise ValueError(f"Invalid MV model {self.config.mv_model}")
        t1 = time.time()
        print(f"Generating multiviews took {t1 - t0:.2f} seconds")

        if debug:
            for i in range(len(multiviews)):
                image = multiviews[i]
                image.save(f'./debug/debug_multiview_{i}.png')

        if upscale_model == 'Aura':
            from .upscalers.pipelines import AuraSRUpscalerPipeline
            upscaler = AuraSRUpscalerPipeline.from_pretrained()
        elif upscale_model == 'NMKD':
            from .upscalers.pipelines import NMKDSiaxUpscalerPipeline
            upscaler = NMKDSiaxUpscalerPipeline.from_pretrained(self.config.device)
        elif upscale_model == 'InvSR':
            from .upscalers.pipelines import InvSRUpscalerPipeline
            upscaler = InvSRUpscalerPipeline.from_pretrained(self.config.device)
        elif upscale_model == 'Flux':
            from .upscalers.pipelines import FluxUpscalerPipeline
            upscaler = FluxUpscalerPipeline.from_pretrained(self.config.device)
        elif upscale_model == 'SD-Upscaler':
            from .utils.imagesuper_utils import Image_Super_Net
            upscaler = Image_Super_Net(self.config.device)
        elif upscale_model == 'Topaz':
            from .upscalers.pipelines import TopazAPIUpscalerPipeline
            upscaler = TopazAPIUpscalerPipeline()
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
                        rgb_img.save(f'./debug/debug_multiview_{i}_upscaled.png')

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
                    multiviews[i].save(f'./debug/debug_multiview_{i}.png')

        roughness_multiviews, metallic_multiviews = [], []
        if pbr:
            from .pbr.pipelines import RGB2XPipeline

            pre_pbr_multiviews = [view.resize((1024, 1024)) for view in multiviews[:6]]

            pbr_pipeline = RGB2XPipeline.from_pretrained("cuda")

            # Do it in batches of 6
            albedo_multiviews, metallic_multiviews, _, roughness_multiviews = (
                self.generate_pbr_for_batch(pbr_pipeline, pre_pbr_multiviews))

            t2 = time.time()
            print(f"PBR texture generation took {t2 - t0:.2f} seconds")

        if self.config.baking_pipeline == 'mv-adapter':
            # Use MV-Adapter for texture baking
            texture_pipeline = TexturePipeline(self.config.mv_adapter_inpaint_weights,
                                               device=self.config.device)
            print('Baking texture with MV-Adapter...')
            t0 = time.time()

            # Lets write all files like MV-Adapter expects
            mv_input_dir = os.path.join(output_dir, 'mvadapter')
            os.makedirs(mv_input_dir, exist_ok=True)

            mesh_path = os.path.join(mv_input_dir, f'tmp.glb')
            mesh.export(mesh_path)

            from .mvadapter.utils.saving import make_image_grid
            mv_path = os.path.join(mv_input_dir, 'multiviews.png')
            make_image_grid(multiviews, rows=1).save(mv_path)

            if len(roughness_multiviews) > 0 and len(metallic_multiviews) > 0:
                from .pbr.pipelines import RGB2XPipeline
                # Combine roughness and metallic textures
                orm_images = []
                for i in range(len(roughness_multiviews)):
                    orm_image = RGB2XPipeline.combine_roughness_metalness(
                        roughness_multiviews[i],
                        metallic_multiviews[i]
                    )
                    orm_images.append(orm_image)

                orm_path = os.path.join(mv_input_dir, 'orm_multiviews.png')
                make_image_grid(orm_images, rows=1).save(orm_path)

                textured_mesh = texture_pipeline(
                    mesh_path=mesh_path,
                    move_to_center=True,
                    save_dir=output_dir,
                    save_name=output_name,
                    uv_unwarp=True,
                    preprocess_mesh=False,
                    uv_size=self.config.texture_size,
                    camera_elevation_deg=selected_camera_elevs,
                    camera_azimuth_deg=selected_camera_azims,
                    base_color_path=mv_path,
                    base_color_process_config=ModProcessConfig(inpaint_mode='view'),
                    orm_path=orm_path,
                    orm_process_config=ModProcessConfig(inpaint_mode='view'),
                    debug_mode=debug
                )
            else:
                textured_mesh = texture_pipeline(
                    mesh_path=mesh_path,
                    move_to_center=True,
                    save_dir=output_dir,
                    save_name=output_name,
                    uv_unwarp=True,
                    preprocess_mesh=False,
                    uv_size=self.config.texture_size,
                    camera_elevation_deg=selected_camera_elevs,
                    camera_azimuth_deg=selected_camera_azims,
                    rgb_path=mv_path,
                    rgb_process_config=ModProcessConfig(inpaint_mode='view'),
                    debug_mode=debug
                )
            t1 = time.time()
            print(f"Texture baking with MV-Adapter took {t1 - t0:.2f} seconds")
        else:
            # Use Hunyuan3D for texture baking
            print('Wrapping UV...')
            t0 = time.time()
            if unwrap_method == 'open3d':
                from .utils.uv_warp_utils import open3d_mesh_uv_wrap
                mesh = open3d_mesh_uv_wrap(mesh, resolution=texture_size)
            elif unwrap_method == 'bpy':
                from .utils.uv_warp_utils import bpy_unwrap_mesh
                mesh = bpy_unwrap_mesh(mesh)
            elif unwrap_method == 'xatlas':
                from .utils.uv_warp_utils import mesh_uv_wrap
                mesh = mesh_uv_wrap(mesh, resolution=texture_size)
            else:
                raise ValueError(f"Invalid unwrap method {unwrap_method}")
            t1 = time.time()
            print(f"UV wrapping took {t1 - t0:.2f} seconds")

            self.render.load_mesh(mesh)

            normal_texture, metallic_roughness_texture, metallic_factor, roughness_factor = None, None, None, None
            if pbr:
                from .pbr.pipelines import RGB2XPipeline

                for i in range(len(roughness_multiviews)):
                    roughness_multiviews[i] = roughness_multiviews[i].resize(
                        (self.config.texture_size, self.config.texture_size))
                    metallic_multiviews[i] = metallic_multiviews[i].resize(
                        (self.config.texture_size, self.config.texture_size))

                    if debug:
                        roughness_multiviews[i].save(f'debug_roughness_multiview_{i}.png')
                        metallic_multiviews[i].save(f'debug_metallic_multiview_{i}.png')

                print('Baking roughness PBR texture...')
                roughness_texture, roughness_mask = self.bake_from_multiview(roughness_multiviews,
                                                                             self.config.candidate_camera_elevs,
                                                                             self.config.candidate_camera_azims,
                                                                             self.config.candidate_view_weights)
                roughness_texture = self.texture_inpaint(roughness_texture, roughness_mask)
                roughness_texture = roughness_texture.cpu().numpy()
                roughness_texture = Image.fromarray((roughness_texture * 255).astype(np.uint8))
                print('Baking metallic PBR texture...')
                metallic_texture, metallic_mask = self.bake_from_multiview(metallic_multiviews,
                                                                           self.config.candidate_camera_elevs,
                                                                           self.config.candidate_camera_azims,
                                                                           self.config.candidate_view_weights)
                metallic_texture = self.texture_inpaint(metallic_texture, metallic_mask)
                metallic_texture = metallic_texture.cpu().numpy()
                metallic_texture = Image.fromarray((metallic_texture * 255).astype(np.uint8))
                metallic_roughness_texture = RGB2XPipeline.combine_roughness_metalness(
                    roughness_texture,
                    metallic_texture
                )

                roughness_factor, metallic_factor = 0.8, 0.8

            print('Baking texture...')
            t0 = time.time()
            texture, mask = self.bake_from_multiview(multiviews,
                                                     selected_camera_elevs, selected_camera_azims,
                                                     selected_view_weights)
            t1 = time.time()
            print(f"Texture baking with Hunyuan3D took {t1 - t0:.2f} seconds")

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
