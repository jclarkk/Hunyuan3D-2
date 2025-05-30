import argparse
import os
import time

import trimesh
from PIL import Image

from hy3dgen.rmbg import RMBGRemover
from hy3dgen.shapegen.utils import normalize_mesh
from hy3dgen.texgen import Hunyuan3DPaintPipeline


def run(args):
    if args.prompt is None and args.image_paths is None:
        raise ValueError("Please provide either a prompt or an image")

    if args.prompt is not None and args.image_paths is not None:
        raise ValueError("Please provide either a prompt or an image, not both")

    if args.remesh_method not in [None, 'im', 'bpt', 'None']:
        raise ValueError("Re-mesh type must be either 'im' or 'bpt'")

    if args.texture_size not in [1024, 2048, 3072, 4096]:
        raise ValueError("Texture size must be one of 1024, 2048, 3072, 4096")

    if args.unwrap_method not in ['xatlas', 'open3d', 'bpy']:
        raise ValueError("Unwrap method must be either 'xatlas', 'open3d' or 'bpy'")

    t0 = time.time()
    # Load mesh
    mesh = trimesh.load_mesh(args.mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    # Reduce face count
    if (args.remesh_method is not None and args.remesh_method != 'None') or len(mesh.faces) > 200000:
        from hy3dgen.shapegen.postprocessors import FaceReducer
        mesh = FaceReducer()(mesh, remesh_method=args.remesh_method, max_facenum=200000)

        # Check if face count is still too high
        if len(mesh.faces) > 200000:
            raise ValueError("Face count must be less than or equal to 200000")

    t1 = time.time()
    print(f"Mesh pre-processing took {t1 - t0:.2f} seconds")

    t2 = time.time()
    # Load models
    texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2',
                                                              mv_model=args.mv_model,
                                                              use_delight=args.use_delight,
                                                              local_files_only=args.local_files_only,
                                                              device='cpu' if args.use_mmgp else 'cuda')

    t2i_pipeline = None
    # Handle MMGP offloading
    if args.use_mmgp:
        from mmgp import offload

        profile = args.profile
        kwargs = {}
        if args.prompt is not None:
            from hy3dgen.text2image import HunyuanDiTPipeline
            t2i_pipeline = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
                                              local_files_only=args.local_files_only,
                                              device='cpu' if args.use_mmgp else 'cuda')
            pipe = offload.extract_models("t2i_worker", t2i_pipeline)
        pipe.update(offload.extract_models("texgen_worker", texture_pipeline))
        texture_pipeline.models["multiview_model"].pipeline.vae.use_slicing = True

        if profile != 1 and profile != 3:
            kwargs["budgets"] = {"*": 2200}

        offload.profile(pipe, profile_no=profile, verboseLevel=args.verbose, **kwargs)
    print('3D Paint pipeline loaded')

    t3 = time.time()
    print(f"Model loading took {t3 - t2:.2f} seconds")

    if args.prompt is not None:
        if t2i_pipeline is None:
            from hy3dgen.text2image import HunyuanDiTPipeline
            t2i_pipeline = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
                                              local_files_only=args.local_files_only,
                                              device='cpu' if args.use_mmgp else 'cuda')

        t2 = time.time()
        image = t2i_pipeline(args.prompt)
        images = [image]
        t3 = time.time()
        print(f"Text to image took {t3 - t2:.2f} seconds")
    else:
        # Only one image supported right now
        images = [Image.open(image_path) for image_path in args.image_paths]

    t4 = time.time()
    # Preprocess the image
    processed_images = []
    if args.mv_model == 'hunyuan3d-paint-v2-0':
        for image in images:
            rmbg_remover = RMBGRemover(local_files_only=args.local_files_only)
            image = rmbg_remover(image)
            processed_images.append(image)
    else:
        processed_images = images

    t5 = time.time()
    print(f"Image processing took {t5 - t4:.2f} seconds")

    # Generate texture
    t6 = time.time()
    mesh = texture_pipeline(
        mesh,
        processed_images,
        unwrap_method=args.unwrap_method,
        upscale_model=args.upscale_model,
        pbr=args.pbr,
        debug=args.debug,
        texture_size=args.texture_size,
        enhance_texture_angles=args.enhance_texture_angles,
        seed=args.seed
    )
    t7 = time.time()
    print(f"Texture generation took {t7 - t6:.2f} seconds")

    os.makedirs(args.output_dir, exist_ok=True)

    # Use mesh file name as output name
    output_name = os.path.splitext(os.path.basename(args.mesh_path))[0] + '_textured'

    mesh = normalize_mesh(mesh)

    mesh.export(os.path.join(args.output_dir, '{}.glb'.format(output_name)))

    print(f"Output saved to {args.output_dir}/{output_name}.glb")
    print(f"Total time taken: {t7 - t0:.2f} seconds")


if __name__ == "__main__":
    # Parse arguments and then call run
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_files_only', action='store_true', help='Use local models only')
    parser.add_argument('--image_paths', type=str, nargs='+', default=None,
                        help='Path to input images. Can specify multiple paths separated by spaces')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt for the image')
    parser.add_argument('--mesh_path', type=str, help='Path to input mesh', required=True)
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('--texture_size', type=int, default=1024, help='Texture size')
    parser.add_argument('--remesh_method', type=str, help='Re-mesh method. Must be either "im" or "bpt" if used.',
                        default=None)
    parser.add_argument('--unwrap_method', type=str,
                        help='UV unwrap method. Must be either "xatlas", "open3d" or "bpy"', default='xatlas')
    parser.add_argument('--use_delight', action='store_true', help='Use Delight model', default=False)
    parser.add_argument('--mv_model', type=str, default='hunyuan3d-paint-v2-0', help='Multiview model to use')
    parser.add_argument('--upscale_model', type=str, default=None, help='Upscale model to use')
    parser.add_argument('--enhance_texture_angles', action='store_true', help='Enhance texture angles', default=False)
    parser.add_argument('--pbr', action='store_true', help='Generate PBR textures', default=False)
    parser.add_argument('--debug', action='store_true', help='Debug mode', default=False)
    parser.add_argument('--use_mmgp', action='store_true', help='Use MMGP for offloading', default=False)
    parser.add_argument('--profile', type=int, default=3)
    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()

    run(args)
