import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
from rembg import remove
from typing import List, Union

from .models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
from .pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from .pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from .schedulers.scheduling_shift_snr import ShiftSNRScheduler
from .utils import get_orthogonal_camera, tensor_to_image


class MVAdapterPipelineWrapper:
    """
    A wrapper for MVAdapterI2MVSDXLPipeline to integrate it into Hunyuan3DPaintPipeline.
    Accepts normal maps, position maps, and camera info, and generates multi-view images.
    Number of views is specified at call time.
    """

    @classmethod
    def from_pretrained(
            cls,
            base_model: str = "lykon/dreamshaper-xl-v2-turbo",
            device: str = "cuda",
            local_files_only: bool = False,
            model_cls=MVAdapterI2MVSDXLPipeline
    ):
        common_kwargs = dict(
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )

        pipe_kwargs = {
            "vae": AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", **common_kwargs)
        }

        if model_cls == MVAdapterI2MVSDXLPipeline:
            pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(base_model, **common_kwargs, **pipe_kwargs)
        elif model_cls == MVAdapterT2MVSDXLPipeline:
            pipe = MVAdapterT2MVSDXLPipeline.from_pretrained(base_model, **common_kwargs, **pipe_kwargs)

        pipe.safety_checker = None

        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0
        )
        pipe.init_custom_adapter(num_views=6, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0)
        pipe.load_custom_adapter(
            "huanngzh/mv-adapter",
            weight_name="mvadapter_ig2mv_sdxl.safetensors",
            torch_dtype=torch.float16,
            local_files_only=local_files_only,
        )

        if device == "cuda":
            pipe.to(device, dtype=torch.float16)

        return cls(pipe, device=device)

    def __init__(self, pipeline: MVAdapterI2MVSDXLPipeline, device: str):
        self.pipeline = pipeline
        self.device = device

    def preprocess_reference_image(self, image: Image.Image, height: int, width: int) -> Image.Image:
        """
        Preprocess image to center it and resize with a grey background visible through transparency,
        using rembg for background removal.
        """
        # Convert image to numpy array
        image_np = np.array(image)
        has_alpha = image_np.shape[-1] == 4

        # Handle cropping for RGBA with alpha, or use full image for RGB
        if has_alpha:
            alpha = image_np[..., 3] > 0
            H, W = alpha.shape
            y, x = np.where(alpha)
            y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
            x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
            image_center = image_np[y0:y1, x0:x1]
        else:
            image_center = image_np
            H, W = image_center.shape[:2]

        # Resize the longer side to 90% of target dimensions
        if H > W:
            W = int(W * (height * 0.9) / H)
            H = int(height * 0.9)
        else:
            H = int(H * (width * 0.9) / W)
            W = int(width * 0.9)

        # Resize the image
        image_center = np.array(Image.fromarray(image_center).resize((W, H), Image.LANCZOS))

        # Use rembg to remove background
        image_with_alpha = np.array(remove(Image.fromarray(image_center)))
        image_center = image_with_alpha  # rembg returns an RGBA image

        # Create output array with grey background (128 = 0.5 in float)
        image_out = np.full((height, width, 4), [128, 128, 128, 255], dtype=np.uint8)

        # Place the centered image
        start_h = (height - H) // 2
        start_w = (width - W) // 2
        image_out[start_h:start_h + H, start_w:start_w + W] = image_center

        # Convert to float for blending
        image_out = image_out.astype(np.float32) / 255.0

        # Apply alpha blending with grey background
        foreground = image_out[:, :, :3] * image_out[:, :, 3:4]
        background = (1 - image_out[:, :, 3:4]) * 0.5
        image_out = foreground + background

        # Convert back to uint8
        image_out = (image_out * 255).clip(0, 255).astype(np.uint8)

        return Image.fromarray(image_out)

    def generate_control_images_from_mesh(self, mesh, num_views, height=768, width=768,
                                          camera_elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
                                          camera_azimuth_deg=[0, 90, 180, 270, 180, 180]):
        """
        Generate control images from a mesh using the original pipeline's approach.
        """
        from .utils.render import NVDiffRastContextWrapper, load_mesh, render

        # Load the mesh
        mesh_copy = mesh.copy()
        current_mesh = load_mesh(mesh_copy, rescale=True, move_to_center=True, device=self.device)

        # Prepare cameras using the same parameters as the original implementation
        cameras = get_orthogonal_camera(
            elevation_deg=camera_elevation_deg,
            distance=[1.8] * num_views,
            left=-0.55,
            right=0.55,
            bottom=-0.55,
            top=0.55,
            azimuth_deg=[x - 90 for x in camera_azimuth_deg],
            device=self.device,
        )

        ctx = NVDiffRastContextWrapper(device=self.device)
        # Render the mesh
        render_out = render(
            ctx,
            current_mesh,
            cameras,
            height=height,
            width=width,
            render_attr=False,
            normal_background=0.0,
        )

        # Extract position and normal maps
        pos_images = tensor_to_image((render_out.pos + 0.5).clamp(0, 1), batched=True)
        normal_images = tensor_to_image((render_out.normal / 2 + 0.5).clamp(0, 1), batched=True)

        pos_tensor = (render_out.pos + 0.5).clamp(0, 1)
        normal_tensor = (render_out.normal / 2 + 0.5).clamp(0, 1)

        control_images = torch.cat([pos_tensor, normal_tensor], dim=-1)
        control_images = control_images.permute(0, 3, 1, 2)
        control_images = control_images.to(device=self.device, dtype=torch.float16)

        return control_images, pos_images, normal_images

    def generate_control_images_from_maps(self, normal_maps, position_maps, height, width):
        """
        Generate control images from Hunyuan's pre-rendered normal and position maps.
        """
        position_tensors = []
        for img in position_maps:
            img_np = np.array(img.resize((width, height))) / 255.0

            pos_tensor = torch.tensor(img_np, dtype=torch.float32) * 2 - 1

            pos_tensor = (pos_tensor + 0.5).clamp(0, 1)

            pos_tensor = pos_tensor.permute(2, 0, 1)
            position_tensors.append(pos_tensor)

        normal_tensors = []
        for img in normal_maps:
            img_np = np.array(img.resize((width, height))) / 255.0

            normal_tensor = torch.tensor(img_np, dtype=torch.float32) * 2 - 1

            normal_tensor = (normal_tensor / 2 + 0.5).clamp(0, 1)

            normal_tensor = normal_tensor.permute(2, 0, 1)
            normal_tensors.append(normal_tensor)

        position_stack = torch.stack(position_tensors, dim=0).to(self.device, dtype=torch.float16)
        normal_stack = torch.stack(normal_tensors, dim=0).to(self.device, dtype=torch.float16)

        control_images = torch.cat([position_stack, normal_stack], dim=1)

        return control_images, position_maps, normal_maps

    @torch.no_grad()
    def __call__(self,
                 mesh,
                 image_prompt: Union[str, Image.Image] = None,
                 normal_maps: List[Image.Image] = None,
                 position_maps: List[Image.Image] = None,
                 camera_elevation_deg: List[int] = [0, 0, 0, 0, 89.99, -89.99],
                 camera_azimuth_deg: List[int] = [0, 90, 180, 270, 180, 180],
                 num_views: int = 6,
                 seed: int = 42,
                 height: int = 1024,
                 width: int = 1024,
                 num_inference_steps: int = 16,
                 guidance_scale: float = 3.0,
                 reference_conditioning_scale: float = 1.0,
                 control_conditioning_scale: float = 1.0,
                 prompt: str = "high quality",
                 negative_prompt: str = "watermark, ugly, deformed, noisy, blurry, low contrast",
                 use_mesh_renderer: bool = True,
                 lora_scale: float = 1.0,
                 batch_size: int = 6,
                 save_debug_images: bool = False):
        """
        Generate multi-view images using the MV-Adapter pipeline.

        Args:
            mesh: Trimesh object if using mesh renderer
            image_prompt: Reference image for conditioning (can be path or PIL Image)
            normal_maps: List of normal maps if not using mesh renderer
            position_maps: List of position maps if not using mesh renderer
            num_views: Number of views to generate
            seed: Random seed for reproducibility
            height: Height of the generated images
            width: Width of the generated images
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for the diffusion model
            reference_conditioning_scale: Scale for the reference image conditioning
            control_conditioning_scale: Scale for the control image conditioning
            prompt: Text prompt for the image generation
            negative_prompt: Negative prompt for the generation
            use_mesh_renderer: Whether to use the mesh renderer or pre-rendered maps
            lora_scale: Scale for LoRA if used
            save_debug_images: Whether to save intermediate images for debugging
        """
        self.pipeline.cond_encoder.to(device='cuda', dtype=torch.float16)
        self.pipeline.to(device='cuda', dtype=torch.float16)

        reference_image = None
        if image_prompt is not None:
            # Prepare reference image
            if isinstance(image_prompt, str):
                reference_image = Image.open(image_prompt)
            elif isinstance(image_prompt, List):
                reference_image = image_prompt[0]
            else:
                reference_image = image_prompt

            reference_image = self.preprocess_reference_image(reference_image, height, width)

            if save_debug_images:
                import os
                debug_dir = "debug_control_images"
                os.makedirs(debug_dir, exist_ok=True)

                reference_image.save(os.path.join(debug_dir, "reference_image.png"))

        # Generate control images
        if use_mesh_renderer and mesh is not None:
            # Use the MV-Adapter mesh rendering approach
            control_images, pos_images, normal_images = self.generate_control_images_from_mesh(
                mesh,
                num_views,
                height,
                width,
                camera_azimuth_deg=camera_azimuth_deg,
                camera_elevation_deg=camera_elevation_deg
            )
            source = "mesh"
        elif normal_maps is not None and position_maps is not None:
            # Use the Hunyuan rendered maps
            control_images, pos_images, normal_images = self.generate_control_images_from_maps(
                normal_maps, position_maps, height, width
            )
            source = "maps"
        else:
            raise ValueError("Either mesh or both normal_maps and position_maps must be provided")

        # Optionally save control images, position maps, and normal maps for visual inspection
        if save_debug_images:
            import os
            debug_dir = "./debug"
            os.makedirs(debug_dir, exist_ok=True)

            pos_tensor = control_images[:, :3, :, :]
            norm_tensor = control_images[:, 3:, :, :]
            print(f"pos_tensor shape: {pos_tensor.shape}, norm_tensor shape: {norm_tensor.shape}")

            for i in range(num_views):
                pos_img_tensor = pos_tensor[i].clamp(0, 1).cpu().float()
                norm_img_tensor = norm_tensor[i].clamp(0, 1).cpu().float()

                # Manual conversion if tensor_to_image fails
                pos_data = (pos_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                norm_data = (norm_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                pos_img = Image.fromarray(pos_data)
                norm_img = Image.fromarray(norm_data)

                pos_img.save(os.path.join(debug_dir, f"control_pos_view_{i}_{source}.png"))
                norm_img.save(os.path.join(debug_dir, f"control_norm_view_{i}_{source}.png"))
                pos_images[i].save(os.path.join(debug_dir, f"orig_pos_view_{i}_{source}.png"))
                normal_images[i].save(os.path.join(debug_dir, f"orig_norm_view_{i}_{source}.png"))

        # Process in batches
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed != -1 else None
        all_images = []

        for batch_start in range(0, num_views, batch_size):
            batch_end = min(batch_start + batch_size, num_views)
            current_num_views = batch_end - batch_start
            batch_control_images = control_images[batch_start:batch_end]

            if reference_image is not None:
                output = self.pipeline(
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=current_num_views,
                    control_image=batch_control_images,
                    control_conditioning_scale=control_conditioning_scale,
                    reference_image=reference_image,
                    reference_conditioning_scale=reference_conditioning_scale,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    cross_attention_kwargs={"scale": lora_scale},
                )
            else:
                output = self.pipeline(
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=current_num_views,
                    control_image=batch_control_images,
                    control_conditioning_scale=control_conditioning_scale,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    cross_attention_kwargs={"scale": lora_scale},
                )
            all_images.extend(output.images)

        if save_debug_images:
            import os
            debug_dir = "debug_output_images"
            os.makedirs(debug_dir, exist_ok=True)
            for i, img in enumerate(all_images):
                img.save(os.path.join(debug_dir, f"output_image_{i}.png"))

        return all_images
