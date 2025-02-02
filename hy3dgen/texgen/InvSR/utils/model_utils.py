import os
import sys
import types
from pathlib import Path

from torchvision.transforms.functional import rgb_to_grayscale

# Create a module for `torchvision.transforms.functional_tensor`
functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = rgb_to_grayscale

# Add this module to sys.modules so other imports can access it
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

from .download_util import load_file_from_url


def load_model(config, num_steps=4, bs=1, tiled_vae=True, color_fix="", chopping_size=128, chopping_bs=8):
    sd_path = "./weights"
    os.makedirs(sd_path, exist_ok=True)
    config.sd_pipe.params.cache_dir = sd_path

    started_ckpt_name = "noise_predictor_sd_turbo_v5.pth"
    started_ckpt_dir = "./weights"
    os.makedirs(started_ckpt_dir, exist_ok=True)
    started_ckpt_path = Path(started_ckpt_dir) / started_ckpt_name
    if not started_ckpt_path.exists():
        load_file_from_url(
            url="https://huggingface.co/OAOA/InvSR/resolve/main/noise_predictor_sd_turbo_v5.pth",
            model_dir=started_ckpt_dir,
            progress=True,
            file_name=started_ckpt_name,
        )

    config.model_start.ckpt_path = str(started_ckpt_path)

    if num_steps == 1:
        config.timesteps = [200, ]
    elif num_steps == 2:
        config.timesteps = [200, 100]
    elif num_steps == 3:
        config.timesteps = [200, 100, 50]
    elif num_steps == 4:
        config.timesteps = [200, 150, 100, 50]
    elif num_steps == 5:
        config.timesteps = [250, 200, 150, 100, 50]
    else:
        assert num_steps <= 5

    config.bs = bs
    config.tiled_vae = tiled_vae
    config.color_fix = color_fix
    config.basesr.chopping.pch_size = chopping_size
    if bs > 1:
        config.basesr.chopping.extra_bs = 1
    else:
        config.basesr.chopping.extra_bs = chopping_bs
