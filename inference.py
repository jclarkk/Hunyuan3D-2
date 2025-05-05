import argparse
import json
import os
import time
from uuid import uuid4

import torch
from PIL import Image
from mmgp import offload

from hy3dgen.rmbg import RMBGRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover, \
    MeshlibCleaner
from hy3dgen.shapegen.utils import normalize_mesh


def run(args):
    if args.face_count > 100000:
        raise ValueError("Face count must be less than or equal to 100000")

    if args.unwrap_method not in ['xatlas', 'open3d', 'bpy']:
        raise ValueError("Unwrap method must be either 'xatlas', 'open3d' or 'bpy'")

    if args.remesh_method not in [None, 'im', 'bpt', 'deepmesh', 'None']:
        raise ValueError("Re-mesh type must be either 'im', 'bpt' or 'deepmesh'")

    t0 = time.time()

    # Preprocess the image
    rmbg_remover = RMBGRemover(local_files_only=args.local_files_only)
    if args.image_json is not None:
        # Read the JSON as dict
        with open(args.image_json) as f:
            image_dict = json.load(f)

        processed_dict = {}
        # Process all images
        for key, val in image_dict.items():
            current_image = Image.open(val)
            current_image = rmbg_remover(current_image, height=args.resolution, width=args.resolution)
            processed_dict[key] = current_image
        image = processed_dict

        # Since texture still relies on first image, we need to ensure it's in context.
        if 'front' in image_dict:
            image_path = image_dict['front']
        else:
            image_path = list(image_dict.values())[0]
    elif args.image_paths is not None:
        # Only one image supported in this argument
        image_path = args.image_paths[0]
        image = Image.open(image_path)
        image = rmbg_remover(image, height=args.resolution, width=args.resolution)

        processed_image_name = os.path.basename(image_path).split('.')[0] + '_input.png'
        image.save(os.path.join(args.output_dir, processed_image_name))
    else:
        raise ValueError("No image paths provided")

    t1 = time.time()
    print(f"Image processing took {t1 - t0:.2f} seconds")

    mc_algo = 'mc' if args.device in ['cpu', 'mps'] else args.mc_algo

    if args.steps is None:
        steps = 5 if 'turbo' in args.geo_model else 30
    else:
        steps = args.steps
    fix_holes = False
    if args.geo_model == 'hunyuan3d-dit-v2-0-turbo':
        mesh_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2',
            use_safetensors=True,
            subfolder='hunyuan3d-dit-v2-0-turbo',
            variant='fp16',
            device=args.device,
            local_files_only=args.local_files_only
        )
    elif args.geo_model == 'hunyuan3d-dit-v2-mini-turbo':
        mesh_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2mini',
            config_path='./configs/hunyuan3d-dit-v2-mini-turbo.yaml',
            use_safetensors=True,
            subfolder='hunyuan3d-dit-v2-mini-turbo',
            variant='fp16',
            device=args.device,
            local_files_only=args.local_files_only
        )
        # In this pipeline we might get holes sometimes
        fix_holes = True
    elif args.geo_model == 'hunyuan3d-dit-v2-mv-turbo':
        mesh_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2mv',
            config_path='./configs/hunyuan3d-dit-v2-mv-turbo.yaml',
            use_safetensors=True,
            subfolder='hunyuan3d-dit-v2-mv-turbo',
            variant='fp16',
            device=args.device,
            local_files_only=args.local_files_only
        )
    else:
        mesh_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2',
            use_safetensors=True,
            device=args.device,
            local_files_only=args.local_files_only
        )
    mesh_pipeline.enable_flashvdm(mc_algo=mc_algo)

    t2 = time.time()
    print('3D DiT pipeline loaded. Took {:.2f} seconds'.format(t2 - t1))

    if args.use_mmgp:
        from hy3dgen.mmgp_utils import replace_property_getter
        # Handle MMGP offloading
        profile = args.profile
        kwargs = {}

        replace_property_getter(mesh_pipeline, "_execution_device", lambda self: "cuda")
        pipe = offload.extract_models("i23d_worker", mesh_pipeline)

    if args.texture:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2',
                                                                  mv_model=args.mv_model,
                                                                  use_delight=args.use_delight,
                                                                  local_files_only=args.local_files_only)
        if args.use_mmgp:
            pipe.update(offload.extract_models("texgen_worker", texture_pipeline))
        texture_pipeline.models["multiview_model"].pipeline.vae.use_slicing = True

    if args.use_mmgp:
        if profile < 5:
            kwargs["pinnedMemory"] = "i23d_worker/model"
        if profile != 1 and profile != 3:
            kwargs["budgets"] = {"*": 2200}

        offload.profile(pipe, profile_no=profile, verboseLevel=args.verbose, **kwargs)

    # Generate mesh
    mesh = mesh_pipeline(image=image,
                         num_inference_steps=steps,
                         octree_resolution=args.octree_resolution,
                         guidance_scale=args.guidance_scale,
                         generator=torch.manual_seed(args.seed))[0]
    t3 = time.time()
    print(f"Mesh generation took {t3 - t2:.2f} seconds")
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    if fix_holes:
        mesh = MeshlibCleaner()(mesh)
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
        print('3D Paint pipeline loaded')
        mesh = texture_pipeline(mesh, image, unwrap_method=args.unwrap_method, seed=args.seed)
        t6 = time.time()
        print(f"Texture generation took {t6 - t5:.2f} seconds")

    os.makedirs(args.output_dir, exist_ok=True)

    # Use image file name as output name
    if image_path is not None:
        output_name = os.path.splitext(os.path.basename(image_path))[0]
    else:
        output_name = str(uuid4()).replace('-', '')

    mesh = normalize_mesh(mesh)

    mesh.export(os.path.join(args.output_dir, '{}.glb'.format(output_name)))

    print(f"Output saved to {args.output_dir}/{output_name}.glb")


if __name__ == "__main__":
    # Parse arguments and then call run
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_files_only', action='store_true', help='Use local models only')
    parser.add_argument('--image_paths', type=str, nargs='+', required=False,
                        help='Path to input images. Can specify multiple paths separated by spaces')
    parser.add_argument('--image_json', type=str, default=None, help='Path to input image json')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('--steps', type=int, default=None, help='Number of inference steps')
    parser.add_argument('--octree_resolution', type=int, default=384)
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--geo_model', type=str, default='hunyuan3d-dit-v2-0')
    parser.add_argument('--mc_algo', type=str, default='dmc')
    parser.add_argument('--remesh_method', type=str,
                        help='Re-mesh method. Must be either "im", "bpt" or "deepmesh" if used.',
                        default=None)
    parser.add_argument('--unwrap_method', type=str,
                        help='UV unwrap method. Must be either "xatlas", "open3d" or "bpy"', default='xatlas')
    parser.add_argument('--face_count', type=int, default=100000, help='Maximum face count for the mesh')
    parser.add_argument('--texture', action='store_true', help='Texture the mesh', default=False)
    parser.add_argument('--resolution', type=int, default=1024, help='Input image resolution (height and width)')
    parser.add_argument('--use_delight', action='store_true', help='Use Delight model', default=False)
    parser.add_argument('--mv_model', type=str, default='hunyuan3d-paint-v2-0', help='Multiview model to use')
    parser.add_argument('--use_mmgp', action='store_true', help='Use MMGP for offloading', default=False)
    parser.add_argument('--profile', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()

    t0 = time.time()
    run(args)
    t1 = time.time()
    print(f"Run time taken: {t1 - t0:.2f} seconds")
