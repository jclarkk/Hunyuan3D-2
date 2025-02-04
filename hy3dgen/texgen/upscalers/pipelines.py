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
        resized_image = input_image.resize((1024, 1024))

        return self.pipe(
            prompt="",
            control_image=resized_image,
            controlnet_conditioning_scale=0.8,
            num_inference_steps=28,
            guidance_scale=3.5,
            height=resized_image.size[1],
            width=resized_image.size[0]
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
        config_path = './hy3dgen/texgen/InvSR/configs/sample-sd-turbo.yaml'
        config = OmegaConf.load(config_path)
        # Load model
        model_utils.load_model(config)
        # Load sampler
        return cls(InvSamplerSR(config), device)

    def __init__(self, sampler, device):
        self.sampler = sampler
        self.device = device

    def __call__(self, input_image: Image.Image) -> Image.Image:
        return Image.fromarray(self.sampler(input_image, self.device))


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
