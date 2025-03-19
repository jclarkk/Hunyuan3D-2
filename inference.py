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
from uuid import uuid4

from hy3dgen.rmbg import RMBGRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline


def run(args):
    if args.face_count > 100000:
        raise ValueError("Face count must be less than or equal to 100000")

    if args.unwrap_method not in ['xatlas', 'open3d', 'bpy']:
        raise ValueError("Unwrap method must be either 'xatlas', 'open3d' or 'bpy'")

    if args.remesh_method not in [None, 'im', 'bpt', 'simplify', 'None']:
        raise ValueError("Re-mesh type must be either 'im', 'bpt' or 'simplify'")

    # Only one image supported right now
    image_path = args.image_paths[0]

    t0 = time.time()

    # Preprocess the image
    image = Image.open(image_path)
    rmbg_remover = RMBGRemover()
    image = rmbg_remover(image, height=1024, width=1024)

    processed_image_name = os.path.basename(image_path).split('.')[0] + '_input.png'
    image.save(os.path.join(args.output_dir, processed_image_name))

    t1 = time.time()
    print(f"Image processing took {t1 - t0:.2f} seconds")

    mc_algo = 'mc' if args.device in ['cpu', 'mps'] else args.mc_algo

    if args.fast:
        mesh_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2',
            use_safetensors=True,
            subfolder='hunyuan3d-dit-v2-0-turbo',
            variant='fp16'
        )
    else:
        mesh_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2',
            use_safetensors=True
        )
    mesh_pipeline.enable_flashvdm(mc_algo=mc_algo)

    t2 = time.time()
    print('3D DiT pipeline loaded. Took {:.2f} seconds'.format(t2 - t1))

    # Generate mesh
    mesh = mesh_pipeline(image=image,
                         num_inference_steps=args.steps,
                         octree_resolution=512,
                         generator=torch.manual_seed(args.seed))[0]
    t3 = time.time()
    print(f"Mesh generation took {t3 - t2:.2f} seconds")
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=args.face_count, remesh_method=args.remesh_method)
    t4 = time.time()
    print(f"Mesh Optimization took {t4 - t3:.2f} seconds")

    # Generate texture
    if args.texture:
        del mesh_pipeline
        torch.cuda.empty_cache()

        # Reload image to use maximum possible resolution for texture model
        image = Image.open(image_path)
        t5 = time.time()
        texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2', mv_model=args.mv_model)
        if args.low_vram_mode:
            texture_pipeline.enable_model_cpu_offload()
            texture_pipeline.models["multiview_model"].pipeline.vae.use_slicing = True
        print('3D Paint pipeline loaded')
        mesh = texture_pipeline(mesh, image=image, unwrap_method=args.unwrap_method, seed=args.seed)
        t6 = time.time()
        print(f"Texture generation took {t6 - t5:.2f} seconds")
    else:
        t6 = t3

    os.makedirs(args.output_dir, exist_ok=True)

    # Use image file name as output name
    if len(args.image_paths) == 1:
        output_name = os.path.splitext(os.path.basename(args.image_paths[0]))[0]
    else:
        output_name = str(uuid4()).replace('-', '')
    mesh.export(os.path.join(args.output_dir, '{}.glb'.format(output_name)))

    print(f"Output saved to {args.output_dir}/{output_name}.glb")
    print(f"Total time taken: {t6 - t0:.2f} seconds")
    sys.stdout.flush()


if __name__ == "__main__":
    # Parse arguments and then call run
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_paths', type=str, nargs='+', required=True,
                        help='Path to input images. Can specify multiple paths separated by spaces')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('--steps', type=int, default=30, help='Number of inference steps')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fast', action='store_true', help='Use fast mode', default=False)
    parser.add_argument('--mc_algo', type=str, default='dmc')
    parser.add_argument('--remesh_method', type=str, help='Re-mesh method. Must be either "im" or "bpt" if used.',
                        default=None)
    parser.add_argument('--unwrap_method', type=str,
                        help='UV unwrap method. Must be either "xatlas", "open3d" or "bpy"', default='xatlas')
    parser.add_argument('--face_count', type=int, default=100000, help='Maximum face count for the mesh')
    parser.add_argument('--texture', action='store_true', help='Texture the mesh', default=False)
    parser.add_argument('--mv_model', type=str, default='hunyuan3d-paint-v2-0', help='Multiview model to use')
    parser.add_argument('--low_vram_mode', action='store_true')

    args = parser.parse_args()

    run(args)
