try:
    # If using Blender's uv unwrap then we must initialize bpy
    import bpy
except ImportError:
    pass

import argparse
import os
import time
import trimesh
from PIL import Image
from mmgp import offload

from hy3dgen.rmbg import RMBGRemover
from hy3dgen.shapegen.postprocessors import import_mesh, reduce_face, FaceReducer
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline


def run(args):
    if args.prompt is None and args.image_paths is None:
        raise ValueError("Please provide either a prompt or an image")

    if args.prompt is not None and args.image_paths is not None:
        raise ValueError("Please provide either a prompt or an image, not both")

    if args.remesh_method not in [None, 'im', 'bpt', 'None']:
        raise ValueError("Re-mesh type must be either 'im' or 'bpt'")

    if args.texture_size not in [1024, 2048]:
        raise ValueError("Texture size must be either 1024 or 2048")

    if args.unwrap_method not in ['xatlas', 'open3d', 'bpy']:
        raise ValueError("Unwrap method must be either 'xatlas', 'open3d' or 'bpy'")

    t0 = time.time()
    # Load mesh
    mesh = trimesh.load_mesh(args.mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    # Reduce face count
    if len(mesh.faces) > 100000:
        ms = import_mesh(mesh)
        ms = reduce_face(ms, max_facenum=90000)
        current_mesh = ms.current_mesh()
        mesh = trimesh.Trimesh(vertices=current_mesh.vertex_matrix(), faces=current_mesh.face_matrix())

    if args.remesh_method is not None and args.remesh_method != 'None':
        mesh = FaceReducer()(mesh, remesh_method=args.remesh_method)

        # Check if face count is still too high
        if len(mesh.faces) > 100000:
            raise ValueError("Face count must be less than or equal to 100000")

    t1 = time.time()
    print(f"Mesh pre-processing took {t1 - t0:.2f} seconds")

    t2 = time.time()
    # Load models
    profile = int(args.profile)
    kwargs = {}
    texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2', mv_model=args.mv_model,
                                                              use_delight=args.use_delight)
    print('3D Paint pipeline loaded')

    pipe = offload.extract_models("texgen_worker", texture_pipeline)
    if args.mv_model == 'hunyuan3d-paint-v2-0':
        texture_pipeline.models["multiview_model"].pipeline.vae.use_slicing = True

    t2i_pipeline = None
    if args.prompt is not None:
        t2i_pipeline = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        pipe.update(offload.extract_models("t2i_worker", t2i_pipeline))

    if profile < 5:
        kwargs["pinnedMemory"] = "texgen_worker/model"
    if profile != 1 and profile != 3:
        kwargs["budgets"] = {"*": 2200}

    offload.profile(pipe, profile_no=profile, verboseLevel=int(args.verbose), **kwargs)
    t3 = time.time()
    print(f"Model loading took {t3 - t2:.2f} seconds")

    if args.prompt is not None:
        t2 = time.time()
        image = t2i_pipeline(args.prompt)
        t3 = time.time()
        print(f"Text to image took {t3 - t2:.2f} seconds")
    else:
        # Only one image supported right now
        image_path = args.image_paths[0]
        image = Image.open(image_path)

    t4 = time.time()
    # Preprocess the image
    rmbg_remover = RMBGRemover()
    image = rmbg_remover(image)

    t5 = time.time()
    print(f"Image processing took {t5 - t4:.2f} seconds")

    # Generate texture
    t6 = time.time()
    mesh = texture_pipeline(
        mesh,
        image=image,
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

    mesh.export(os.path.join(args.output_dir, '{}.glb'.format(output_name)))

    print(f"Output saved to {args.output_dir}/{output_name}.glb")
    print(f"Total time taken: {t7 - t0:.2f} seconds")


if __name__ == "__main__":
    # Parse arguments and then call run
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--profile', type=str, default="3")
    parser.add_argument('--verbose', type=str, default="1")

    args = parser.parse_args()

    run(args)
