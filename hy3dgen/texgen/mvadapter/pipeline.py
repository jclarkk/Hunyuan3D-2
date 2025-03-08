import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from typing import List

from .pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from .schedulers.scheduling_shift_snr import ShiftSNRScheduler


class MVAdapterPipelineWrapper:
    """
    A wrapper for MVAdapterI2MVSDXLPipeline to integrate it into Hunyuan3DPaintPipeline.
    Accepts normal maps, position maps, and camera info, and generates multi-view images.
    Number of views is specified at call time.
    """

    @classmethod
    def from_pretrained(cls, base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                        adapter_path: str = "huanngzh/mv-adapter", device: str = "cuda"):
        """
        Initialize the MV-Adapter pipeline from pretrained weights without specifying num_views.
        """
        # Load the base components, mimicking standalone code
        pipe_kwargs = {
            'vae': AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix"),
        }

        pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(base_model, **pipe_kwargs)

        # Replace scheduler with ShiftSNRScheduler, as in standalone
        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=DDPMScheduler,
        )

        pipe.init_custom_adapter(num_views=6)
        pipe.load_custom_adapter(adapter_path, weight_name="mvadapter_ig2mv_sdxl.safetensors")

        pipe.to(device=device, dtype=torch.float16)
        pipe.cond_encoder.to(device=device, dtype=torch.float16)

        return cls(pipe, device=device)

    def __init__(self, pipeline: MVAdapterI2MVSDXLPipeline, device: str):
        self.pipeline = pipeline
        self.device = device
        self.current_num_views = None

    def preprocess_inputs(self, normal_maps: List[Image.Image], position_maps: List[Image.Image],
                          camera_info: List[int], image_prompt: Image.Image, num_views: int,
                          height: int = 768, width: int = 768):
        """
        Preprocess the inputs to match MVAdapterI2MVSDXLPipeline expectations.
        """
        assert len(normal_maps) == num_views, f"Expected {num_views} normal maps, got {len(normal_maps)}"
        assert len(position_maps) == num_views, f"Expected {num_views} position maps, got {len(position_maps)}"
        assert len(camera_info) == num_views, f"Expected {num_views} camera info entries, got {len(camera_info)}"

        normal_tensors = [torch.tensor(np.array(img.resize((width, height))) / 255.0).permute(2, 0, 1).float()
                          for img in normal_maps]
        position_tensors = [torch.tensor(np.array(img.resize((width, height))) / 255.0).permute(2, 0, 1).float()
                            for img in position_maps]

        normal_stack = torch.stack(normal_tensors, dim=0).to(self.device, dtype=torch.float16)
        position_stack = torch.stack(position_tensors, dim=0).to(self.device, dtype=torch.float16)
        control_images = torch.cat([position_stack, normal_stack], dim=1)  # [num_views, 6, H, W]

        reference_image = image_prompt.resize((width, height))
        if reference_image.mode == "RGBA":
            reference_image = self._preprocess_rgba(reference_image, height, width)

        return control_images, reference_image

    def _preprocess_rgba(self, image: Image.Image, height: int, width: int) -> Image.Image:
        """
        Preprocess RGBA image to remove background and center it (matches standalone preprocess_image).
        """
        image_np = np.array(image)
        alpha = image_np[..., 3] > 0
        H, W = alpha.shape
        y, x = np.where(alpha)
        y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
        x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
        image_center = image_np[y0:y1, x0:x1]
        H, W, _ = image_center.shape
        if H > W:
            W = int(W * (height * 0.9) / H)
            H = int(height * 0.9)
        else:
            H = int(H * (width * 0.9) / W)
            W = int(width * 0.9)
        image_center = np.array(Image.fromarray(image_center).resize((W, H)))
        start_h = (height - H) // 2
        start_w = (width - W) // 2
        image_out = np.zeros((height, width, 4), dtype=np.uint8)
        image_out[start_h:start_h + H, start_w:start_w + W] = image_center
        image_out = image_out.astype(np.float32) / 255.0
        image_out = image_out[:, :, :3] * image_out[:, :, 3:4] + (1 - image_out[:, :, 3:4]) * 0.5
        image_out = (image_out * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image_out)



    @torch.no_grad()
    def __call__(self, image_prompt: Image.Image, normal_maps: List[Image.Image],
                 position_maps: List[Image.Image], camera_info: List[int], num_views: int,
                 seed: int = 42, height: int = 768, width: int = 768, num_inference_steps: int = 50,
                 guidance_scale: float = 3.0,
                 negative_prompt: str = "watermark, ugly, deformed, noisy, blurry, low contrast"):

        if self.current_num_views != num_views:
            self.pipeline.init_custom_adapter(num_views=num_views)
            self.pipeline.to(device=self.device, dtype=torch.float16)
            self.pipeline.cond_encoder.to(device=self.device, dtype=torch.float16)
            self.current_num_views = num_views

        control_images, reference_image = self.preprocess_inputs(
            normal_maps, position_maps, camera_info, image_prompt, num_views, height, width
        )

        generator = torch.Generator(device=self.device).manual_seed(seed) if seed != -1 else None

        images = self.pipeline(
            prompt="high quality",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_views,
            control_image=control_images,
            control_conditioning_scale=1.0,
            reference_image=reference_image,
            reference_conditioning_scale=1.0,
            negative_prompt=negative_prompt,
            generator=generator,
            output_type="pil"
        ).images

        return images