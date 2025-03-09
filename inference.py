try:
    # If using Blender's uv unwrap then we must initialize bpy
    import bpy
except ImportError:
    pass

import argparse
import os
import sys
import time
import torch
from PIL import Image
from mmgp import offload
from uuid import uuid4

from hy3dgen.rmbg import RMBGRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline


def run(args):
    if args.face_count > 100000:
        raise ValueError("Face count must be less than or equal to 100000")

    if args.unwrap_method not in ['xatlas', 'open3d', 'bpy']:
        raise ValueError("Unwrap method must be either 'xatlas', 'open3d' or 'bpy'")

    if args.remesh_method not in [None, 'im', 'bpt', 'None']:
        raise ValueError("Re-mesh type must be either 'im' or 'bpt'")

    # Only one image supported right now
    image_path = args.image_paths[0]

    t0 = time.time()

    # Preprocess the image
    image = Image.open(image_path)
    rmbg_remover = RMBGRemover()
    image = rmbg_remover(image)

    processed_image_name = os.path.basename(image_path).split('.')[0] + '_input.png'
    image.save(os.path.join(args.output_dir, processed_image_name))

    t1 = time.time()
    print(f"Image processing took {t1 - t0:.2f} seconds")

    if args.fast:
        mesh_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2',
            use_safetensors=True,
            subfolder='hunyuan3d-dit-v2-0-fast',
            variant='fp16'
        )
    else:
        mesh_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2',
            use_safetensors=True
        )
    print('3D DiT pipeline loaded')

    profile = int(args.profile)
    kwargs = {}
    pipe = offload.extract_models("mesh_worker", mesh_pipeline)

    if args.texture:
        texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2', mv_model=args.mv_model)
        print('3D Paint pipeline loaded')

        pipe.update(offload.extract_models("texgen_worker", texture_pipeline))
        texture_pipeline.models["multiview_model"].pipeline.vae.use_slicing = True

    if profile < 5:
        kwargs["pinnedMemory"] = "i23d_worker/model"
    if profile != 1 and profile != 3:
        kwargs["budgets"] = {"*": 2200}

    offload.profile(pipe, profile_no=profile, verboseLevel=int(args.verbose), **kwargs)

    # Generate mesh
    t2 = time.time()
    mesh = mesh_pipeline(image=image, num_inference_steps=30, mc_algo='mc',
                         generator=torch.manual_seed(args.seed))[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=args.face_count, remesh_method=args.remesh_method)
    t3 = time.time()
    print(f"Mesh generation took {t3 - t2:.2f} seconds")

    # Generate texture
    if args.texture:
        del mesh_pipeline
        torch.cuda.empty_cache()

        # Reload image to use maximum possible resolution for texture model
        image = Image.open(image_path)
        t4 = time.time()
        mesh = texture_pipeline(mesh, image=image, unwrap_method=args.unwrap_method, seed=args.seed)
        t5 = time.time()
        print(f"Texture generation took {t5 - t4:.2f} seconds")
    else:
        t5 = t3

    os.makedirs(args.output_dir, exist_ok=True)

    # Use image file name as output name
    if len(args.image_paths) == 1:
        output_name = os.path.splitext(os.path.basename(args.image_paths[0]))[0]
    else:
        output_name = str(uuid4()).replace('-', '')
    mesh.export(os.path.join(args.output_dir, '{}.glb'.format(output_name)))

    print(f"Output saved to {args.output_dir}/{output_name}.glb")
    print(f"Total time taken: {t5 - t0:.2f} seconds")
    sys.stdout.flush()


if __name__ == "__main__":
    # Parse arguments and then call run
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_paths', type=str, nargs='+', required=True,
                        help='Path to input images. Can specify multiple paths separated by spaces')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('--fast', action='store_true', help='Use fast mode', default=False)
    parser.add_argument('--remesh_method', type=str, help='Re-mesh method. Must be either "im" or "bpt" if used.',
                        default=None)
    parser.add_argument('--unwrap_method', type=str,
                        help='UV unwrap method. Must be either "xatlas", "open3d" or "bpy"', default='xatlas')
    parser.add_argument('--face_count', type=int, default=100000, help='Maximum face count for the mesh')
    parser.add_argument('--texture', action='store_true', help='Texture the mesh', default=False)
    parser.add_argument('--mv_model', type=str, default='hunyuan3d-paint-v2-0', help='Multiview model to use')
    parser.add_argument('--profile', type=str, default="3")
    parser.add_argument('--verbose', type=str, default="1")

    args = parser.parse_args()

    run(args)
