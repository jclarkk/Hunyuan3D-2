import numpy as np
import torch
import trimesh
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.utils.tensor_fragment import fragment_address
from torch.serialization import add_safe_globals

from .model import data_utils
from .model.model import MeshTransformer
from .model.serializaiton import BPT_deserialize
from .utils import sample_pc, joint_filter


class BPTPipeline:

    @classmethod
    def from_pretrained(cls):
        add_safe_globals([LossScaler, fragment_address, ZeroStageEnum])

        model = MeshTransformer()
        model.load('./weights/bpt-8-16-500m.pt')
        model = model.eval()
        model = model.half()
        model = model.cuda()

        return cls(model)

    def __init__(self, model):
        self.model = model

    def __call__(self, mesh: trimesh.Trimesh):
        # Convert mesh to point cloud
        pc_normal = sample_pc(mesh, pc_num=8192, with_normal=True)
        pc_normal = pc_normal[None, :, :] if len(pc_normal.shape) == 2 else pc_normal

        pc_tensor = torch.from_numpy(pc_normal).cuda().half()
        if len(pc_tensor.shape) == 2:
            pc_tensor = pc_tensor.unsqueeze(0)

        codes = self.model.generate(
            pc=pc_tensor,
            filter_logits_fn=joint_filter,
            filter_kwargs=dict(k=50, p=0.95),
            return_codes=True,
        )

        coords = []
        try:
            for i in range(len(codes)):
                code = codes[i]
                code = code[code != self.model.pad_id].cpu().numpy()
                vertices = BPT_deserialize(
                    code,
                    block_size=self.model.block_size,
                    offset_size=self.model.offset_size,
                    use_special_block=self.model.use_special_block,
                )
                coords.append(vertices)
        except:
            coords.append(np.zeros(3, 3))

        # convert coordinates to mesh
        vertices = coords[0]
        faces = torch.arange(1, len(vertices) + 1).view(-1, 3)

        # Move to CPU
        faces = faces.cpu().numpy()

        return data_utils.to_mesh(vertices, faces, transpose=False, post_process=True)
