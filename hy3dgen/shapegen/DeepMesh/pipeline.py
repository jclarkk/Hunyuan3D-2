import os

import numpy as np
import torch
import trimesh
from torch.nn.parallel import DistributedDataParallel as DDP

from .lit_gpt.config import Config
from .lit_gpt.model_cache import GPTCache
from .sample import ar_sample_kvcache, setup_distributed_mode
from .sft.datasets.data_utils import to_mesh
from .sft.datasets.serializaiton import deserialize


class DeepMeshPipeline:

    @classmethod
    def from_pretrained(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        setup_distributed_mode(0, 1)
        model_name = f"Diff_LLaMA_551M"
        config = Config.from_name(model_name)
        config.padded_vocab_size = (2 * 4 ** 3) + (8 ** 3) + (16 ** 3) + 1 + 1
        config.block_size = 270000
        model = GPTCache(config).to('cuda')
        loaded_state = torch.load('./weights/deepmesh.bin', weights_only=False)
        model.load_state_dict(loaded_state, strict=False)
        return cls(model)

    def __init__(self, model):
        self.model = model

    def __call__(self, mesh: trimesh.Trimesh, pc_num=16384):
        # Convert mesh to point cloud
        points, face_idx = mesh.sample(50000, return_index=True)
        normals = mesh.face_normals[face_idx]
        pc_normal = np.concatenate([points[:, [2, 0, 1]], normals[:, [2, 0, 1]]], axis=-1, dtype=np.float16)
        ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
        pc_normal = pc_normal[ind]

        repeat_num = 4
        temperature = 0.5
        steps = 90000
        cond_pc = torch.tensor(pc_normal, dtype=torch.float16, device='cuda')
        prompt = torch.tensor([[4736]], dtype=torch.long, device='cuda').repeat(repeat_num, 1)

        # Run inference
        output_ids, _ = ar_sample_kvcache(self.model,
                                          prompt=prompt,
                                          pc=cond_pc.repeat(repeat_num, 1, 1),
                                          window_size=9000,
                                          temperature=temperature,
                                          context_length=steps,
                                          device='cuda')

        # Post-process first result
        code = output_ids[0][1:]
        index = (code >= 4737).nonzero()
        if index.numel() > 0:
            code = code[:index[0, 0].item()].cpu().numpy().astype(np.int64)
        else:
            code = code.cpu().numpy().astype(np.int64)
        vertices = deserialize(code)
        if len(vertices) == 0:
            print("Empty vertices:", len(vertices))
            return None
        vertices = vertices[..., [2, 1, 0]]
        faces = torch.arange(1, len(vertices) + 1, device='cuda').view(-1, 3)
        return to_mesh(vertices, faces, transpose=False, post_process=True)
