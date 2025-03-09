import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from typing import List, Union

from .models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
from .pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from .schedulers.scheduling_shift_snr import ShiftSNRScheduler
from .utils import get_orthogonal_camera, tensor_to_image, make_image_grid
from .utils.render import NVDiffRastContextWrapper, load_mesh, render


class MVAdapterPipelineWrapper:
    """
    A wrapper for MVAdapterI2MVSDXLPipeline to integrate it into Hunyuan3DPaintPipeline.
    Accepts normal maps, position maps, and camera info, and generates multi-view images.
    Number of views is specified at call time.
    """

    @classmethod
    def from_pretrained(cls, base_model: str = "stabilityai/stable-diffusion-xl-base-1.0", device: str = "cuda"):
        """
        Initialize the MV-Adapter pipeline from pretrained weights without specifying num_views.
        """
        pipe_kwargs = {
            'vae': AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix"),
        }
        pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(base_model, **pipe_kwargs)
        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=DDIMScheduler
        )
        pipe.to(device=device, dtype=torch.float16)
        pipe.cond_encoder.to(device=device, dtype=torch.float16)
        return cls(pipe, device=device)

    def __init__(self, pipeline: MVAdapterI2MVSDXLPipeline, device: str):
        self.pipeline = pipeline
        self.device = device
        self.ctx = NVDiffRastContextWrapper(device=device)

    def preprocess_reference_image(self, image: Image.Image, height: int, width: int) -> Image.Image:
        """
        Preprocess RGBA image to remove background and center it (matches standalone preprocess_image).
        """
        if image.mode != "RGBA":
            image = image.convert("RGB")
            return image.resize((width, height))

        image_np = np.array(image)
        alpha = image_np[..., 3] > 0
        H, W = alpha.shape

        # Get the bounding box of alpha
        y, x = np.where(alpha)
        y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
        x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
        image_center = image_np[y0:y1, x0:x1]

        # Resize the longer side to H * 0.9
        H, W = image_center.shape[:2]
        if H > W:
            W = int(W * (height * 0.9) / H)
            H = int(height * 0.9)
        else:
            H = int(H * (width * 0.9) / W)
            W = int(width * 0.9)

        image_center = np.array(Image.fromarray(image_center).resize((W, H)))

        # Pad to H, W
        start_h = (height - H) // 2
        start_w = (width - W) // 2
        image_out = np.zeros((height, width, 4), dtype=np.uint8)
        image_out[start_h:start_h + H, start_w:start_w + W] = image_center

        # Apply alpha blending with gray background
        image_out = image_out.astype(np.float32) / 255.0
        image_out = image_out[:, :, :3] * image_out[:, :, 3:4] + (1 - image_out[:, :, 3:4]) * 0.5
        image_out = (image_out * 255).clip(0, 255).astype(np.uint8)

        return Image.fromarray(image_out)

    def generate_control_images_from_mesh(self, mesh, num_views, height=768, width=768):
        """
        Generate control images from a mesh using the original pipeline's approach.
        """
        # Load the mesh
        mesh = load_mesh(mesh, rescale=True, device=self.device)

        # Prepare cameras using the same parameters as the original implementation
        cameras = get_orthogonal_camera(
            elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
            distance=[1.8] * num_views,
            left=-0.55,
            right=0.55,
            bottom=-0.55,
            top=0.55,
            azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
            device=self.device,
        )

        # Render the mesh
        render_out = render(
            self.ctx,
            mesh,
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
        print(f"Combined tensor shape: {control_images.shape}")

        control_images = control_images.permute(0, 3, 1, 2)
        print(f"Permuted tensor shape: {control_images.shape}")

        control_images = control_images.to(device=self.device, dtype=torch.float16)

        return control_images, pos_images, normal_images

    def generate_control_images_from_maps(self, normal_maps, position_maps, num_views, height, width):
        """
        Generate control images from Hunyuan's pre-rendered normal and position maps.
        """
        assert len(normal_maps) == num_views, f"Expected {num_views} normal maps, got {len(normal_maps)}"
        assert len(position_maps) == num_views, f"Expected {num_views} position maps, got {len(position_maps)}"

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
                 camera_info: List[int] = None,
                 num_views: int = 6,
                 seed: int = 42,
                 height: int = 768,
                 width: int = 768,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 3.0,
                 reference_conditioning_scale: float = 1.0,
                 control_conditioning_scale: float = 1.0,
                 prompt: str = "high quality",
                 negative_prompt: str = "watermark, ugly, deformed, noisy, blurry, low contrast",
                 use_mesh_renderer: bool = False,
                 lora_scale: float = 1.0,
                 save_debug_images: bool = True):
        """
        Generate multi-view images using the MV-Adapter pipeline.

        Args:
            mesh: Trimesh object if using mesh renderer
            image_prompt: Reference image for conditioning (can be path or PIL Image)
            normal_maps: List of normal maps if not using mesh renderer
            position_maps: List of position maps if not using mesh renderer
            camera_info: List of camera information if not using mesh renderer
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

        # Initialize the pipeline with the custom adapter since we can only declate num of views at call-time.
        self.pipeline.init_custom_adapter(num_views=num_views,
                                          self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0)

        self.pipeline.load_custom_adapter('huanngzh/mv-adapter', weight_name="mvadapter_ig2mv_sdxl.safetensors")
        self.pipeline.to(device=self.device, dtype=torch.float16)
        self.pipeline.cond_encoder.to(device=self.device, dtype=torch.float16)
        self.current_num_views = num_views

        # Prepare reference image
        if isinstance(image_prompt, str):
            reference_image = Image.open(image_prompt)
        else:
            reference_image = image_prompt

        reference_image = self.preprocess_reference_image(reference_image, height, width)

        # Generate control images
        if use_mesh_renderer and mesh is not None:
            # Use the MV-Adapter mesh rendering approach
            control_images, pos_images, normal_images = self.generate_control_images_from_mesh(
                mesh, num_views, height, width
            )
        elif normal_maps is not None and position_maps is not None:
            # Use the Hunyuan rendered maps
            control_images, pos_images, normal_images = self.generate_control_images_from_maps(
                normal_maps, position_maps, num_views, height, width
            )
        else:
            raise ValueError("Either mesh_path or both normal_maps and position_maps must be provided")

        generator = torch.Generator(device=self.device).manual_seed(seed) if seed != -1 else None

        # Run the pipeline
        output = self.pipeline(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_views,
            control_image=control_images,
            control_conditioning_scale=control_conditioning_scale,
            reference_image=reference_image,
            reference_conditioning_scale=reference_conditioning_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            cross_attention_kwargs={"scale": lora_scale},
        )

        images = output.images

        return images
