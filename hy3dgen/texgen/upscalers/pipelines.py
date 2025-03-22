import numpy as np
import torch
from PIL import Image


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


class RealESRGANUpscalerPipeline:
    """
        High quality upscaling with good performance using Real-ESRGAN
    """

    @classmethod
    def from_pretrained(cls, device):
        from realesrgan import RealESRGAN
        # Initialize Real-ESRGAN with 4x upscaling model
        model = RealESRGAN(device=device, scale=4)
        model.load_weights('weights/RealESRGAN_x4plus.pth', download=True)
        return cls(model, device)

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, input_image: Image.Image) -> Image.Image:
        # Convert PIL Image to numpy array
        input_array = np.array(input_image)

        # Perform upscaling
        upscaled_array = self.model.predict(input_array)

        # Convert back to PIL Image
        return Image.fromarray(upscaled_array)


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
