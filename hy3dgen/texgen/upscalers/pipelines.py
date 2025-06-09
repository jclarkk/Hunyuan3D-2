import numpy as np
import os
import requests
import time
import torch
from PIL import Image
from io import BytesIO


class FluxUpscalerPipeline:
    """
        Highest quality but slow
    """

    @classmethod
    def from_pretrained(cls, device):
        from diffusers import FluxControlNetModel
        from diffusers.pipelines import FluxControlNetPipeline
        # Load pipeline
        controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler",
            torch_dtype=torch.bfloat16
        )
        pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()

        return cls(pipe, device)

    def __init__(self, pipe, device):
        self.pipe = pipe
        self.device = device

    def __call__(self, input_image: Image.Image) -> Image.Image:
        w, h = input_image.size
        input_image = input_image.resize((w * 4, h * 4))

        return self.pipe(
            prompt="",
            control_image=input_image,
            controlnet_conditioning_scale=0.8,
            num_inference_steps=28,
            guidance_scale=3.5,
            height=input_image.size[1],
            width=input_image.size[0]
        ).images[0]


class InvSRUpscalerPipeline:
    """
        High quality but can create artifacts
    """

    @classmethod
    def from_pretrained(cls, device):
        from omegaconf import OmegaConf
        from .InvSR.sampler_invsr import InvSamplerSR
        from .InvSR.utils import model_utils
        # Load config
        config_path = './hy3dgen/texgen/upscalers/InvSR/configs/sample-sd-turbo.yaml'
        config = OmegaConf.load(config_path)
        # Load model
        model_utils.load_model(config)
        # Load sampler
        return cls(InvSamplerSR(config), device)

    def __init__(self, sampler, device):
        self.sampler = sampler
        self.device = device

    def __call__(self, input_image: Image.Image) -> Image.Image:
        from .InvSR.sampler_invsr import process_image, pil_to_tensor
        input_tensor = pil_to_tensor(input_image)
        return process_image(self.sampler, input_tensor)


class NMKDSiaxUpscalerPipeline:
    """
        High quality upscaling with good performance using NMKD
    """

    @classmethod
    def from_pretrained(cls, device):
        from spandrel import ModelLoader
        # Initialize Real-ESRGAN with 4x upscaling model
        model = ModelLoader().load_from_file('./weights/4x_NMKD-Siax_200k.pth')
        model.to(device).eval().half()
        return cls(model, device)

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, input_image: Image.Image) -> Image.Image:
        from torchvision import transforms

        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        input_tensor = to_tensor(input_image).unsqueeze(0).to(self.device).half()

        with torch.no_grad():
            output = self.model(input_tensor).float().clamp(0, 1).squeeze(0).cpu()

        return to_pil(output)


class AuraSRUpscalerPipeline:
    """
        Medium quality but fast
    """

    @classmethod
    def from_pretrained(cls):
        from aura_sr import AuraSR
        return cls(AuraSR.from_pretrained("fal/AuraSR-v2"))

    def __init__(self, pipe):
        self.pipe = pipe

    def __call__(self, input_image: Image.Image) -> Image.Image:
        return self.pipe.upscale_4x_overlapped(input_image, max_batch_size=16)


class TopazAPIUpscalerPipeline:
    """
    High quality upscaling using Topaz synchronous API.
    """

    def __init__(self, mode: str = 'enhance'):
        self.topaz_api_key = os.getenv('TOPAZ_API_KEY')
        self.topaz_url = 'https://api.topazlabs.com/image/v1/enhance'
        self.output_height = 4096
        self.output_width = 4096
        self.model = 'Standard V2'
        self.output_format = 'png'
        self.max_retries = 5
        self.backoff_base = 2

    def __call__(self, input_image: Image.Image) -> Image.Image:
        image_bytes = BytesIO()
        input_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        headers = {
            'X-API-Key': self.topaz_api_key,
            'accept': f'image/{self.output_format}',
        }

        files = {
            'image': ('input.png', image_bytes, 'image/png')
        }

        data = {
            'model': self.model,
            'output_height': self.output_height,
            'output_width': self.output_width,
            'output_format': self.output_format
        }

        for attempt in range(self.max_retries):
            response = requests.post(self.topaz_url, headers=headers, files=files, data=data)

            if response.status_code == 200:
                return Image.open(BytesIO(response.content))

            elif response.status_code == 429:
                sleep_time = self.backoff_base ** attempt
                time.sleep(sleep_time)
                continue

            else:
                response.raise_for_status()

        raise Exception("Topaz sync upscaling failed after retries.")
