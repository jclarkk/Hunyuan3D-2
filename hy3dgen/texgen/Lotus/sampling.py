import numpy as np
import torch
from PIL import Image

from .pipeline import LotusGPipeline


class LotusSampler:

    def __init__(self, device):
        self.device = device
        self.pipeline = LotusGPipeline.from_pretrained(
            "jingheya/lotus-normal-g-v1-1",
            torch_dtype=torch.float32,
        )
        self.pipeline = self.pipeline.to(device)

    def __call__(self, input_image: [Image.Image]):
        """Convert an image to normal"""
        test_image = np.array(input_image).astype(np.float32)
        test_image = torch.tensor(test_image).permute(2, 0, 1).unsqueeze(0)
        test_image = test_image / 127.5 - 1.0
        test_image = test_image.to(self.device)

        task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(self.device)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

        # Run
        pred = self.pipeline(
            rgb_in=test_image,
            prompt='',
            num_inference_steps=1,
            output_type='pil',
            timesteps=[999],
            task_emb=task_emb,
            processing_res=0,
            match_input_res=True
        ).images

        return pred
