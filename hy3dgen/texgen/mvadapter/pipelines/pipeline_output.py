from dataclasses import dataclass
from typing import Optional


@dataclass
class TexturePipelineOutput:
    shaded_model_save_path: Optional[str] = None
    pbr_model_save_path: Optional[str] = None
