import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler

from .rgb2x_pipeline import StableDiffusionAOVMatEstPipeline


class RGB2XPipeline:

    @classmethod
    def from_pretrained(cls, device):
        pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
            "zheng95z/rgb-to-x",
            torch_dtype=torch.float16
        ).to(device)
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
        )
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)

        return cls(pipe)

    def __init__(self, pipe):
        self.pipe = pipe

    def __call__(self, input_image: Image.Image, resolution=1024) -> dict:
        generator = torch.Generator(device="cuda").manual_seed(42)

        required_aovs = ["albedo", "normal", "roughness", "metallic"]
        prompts = {
            "albedo": "Albedo (diffuse basecolor)",
            "normal": "Camera-space Normal",
            "roughness": "Roughness",
            "metallic": "Metallicness"
        }

        pbr_dict = {}
        for aov_name in required_aovs:
            prompt = prompts[aov_name]
            generated_image = self.pipe(
                prompt=prompt,
                photo=input_image,
                num_inference_steps=50,
                height=resolution,
                width=resolution,
                generator=generator,
                required_aovs=[aov_name],
            ).images[0][0]

            pbr_dict[aov_name] = generated_image

        return pbr_dict

    @staticmethod
    def combine_roughness_metalness(roughness_map: Image.Image, metalness_map: Image.Image) -> Image.Image:
        """Pack roughness into G channel and metalness into B channel."""
        metalness = np.array(metalness_map)
        roughness = np.array(roughness_map)

        # Convert grayscale to single channel
        if metalness.ndim == 3:
            metalness = metalness[..., 0]
        if roughness.ndim == 3:
            roughness = roughness[..., 0]

        # Convert to 8-bit range if necessary
        if metalness.max() <= 1:
            metalness = (metalness * 255).astype(np.uint8)
        if roughness.max() <= 1:
            roughness = (roughness * 255).astype(np.uint8)

        # Pack into R=0, G=roughness, B=metalness
        height, width = roughness.shape
        zero_channel = np.zeros((height, width), dtype=np.uint8)
        packed_image = np.stack([zero_channel, roughness, metalness], axis=-1)

        return Image.fromarray(packed_image, mode='RGB')

    @staticmethod
    def analyze_texture(texture_np):
        """
        Analyzes a texture image to estimate roughness and metallic properties dynamically.
        """
        contrast = np.std(texture_np)

        # Normalize to 0-1
        metallic_likelihood = contrast / 255.0

        roughness_likelihood = 1.0 - metallic_likelihood

        return metallic_likelihood, roughness_likelihood
