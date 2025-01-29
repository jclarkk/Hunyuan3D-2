import argparse
import os
import time
import torch.cuda
import trimesh
from mmgp import offload
from PIL import Image

from hy3dgen.rmbg import preprocess_image
from hy3dgen.shapegen.postprocessors import import_mesh, reduce_face
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline


def run(args):
    if args.prompt is None and args.image_paths is None:
        raise ValueError("Please provide either a prompt or an image")

    if args.prompt is not None and args.image_paths is not None:
        raise ValueError("Please provide either a prompt or an image, not both")

    if args.texture_size not in [2048, 4096]:
        raise ValueError("Texture size must either be 2k or 4k")

    t2i_pipeline = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled', use_mmgp=args.mmgp)
    texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2', use_mmgp=args.mmgp)

    if args.mmgp:
        # Handle MMGP offloading
        profile = args.mmgp_profile
        kwargs = {}

        pipe = offload.extract_models("t2i_worker", t2i_pipeline)
        pipe.update(offload.extract_models("texgen_worker", texture_pipeline))
        texture_pipeline.models["multiview_model"].pipeline.vae.use_slicing = True

        if profile != 1 and profile != 3:
            kwargs["budgets"] = {"*": 2200}

        offload.profile(pipe, profile_no=profile, verboseLevel=args.mmgp_verbose, **kwargs)

    if args.prompt is not None:
        t0 = time.time()
        image = t2i_pipeline(args.prompt)
        t1 = time.time()
        print(f"Text to image took {t1 - t0:.2f} seconds")
    else:
        # Only one image supported right now
        image_path = args.image_paths[0]
        image = Image.open(image_path)

    t0 = time.time()

    # Preprocess the image
    image = preprocess_image(image)

    t1 = time.time()
    print(f"Image processing took {t1 - t0:.2f} seconds")

    # Load mesh
    mesh = trimesh.load_mesh(args.mesh_path)
    mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    # Reduce face count
    if len(mesh.faces) > 100000:
        ms = import_mesh(mesh)
        ms = reduce_face(ms, max_facenum=100000)
        current_mesh = ms.current_mesh()
        mesh = trimesh.Trimesh(vertices=current_mesh.vertex_matrix(), faces=current_mesh.face_matrix())

    # Generate texture
    t4 = time.time()
    mesh = texture_pipeline(
        mesh,
        image=image,
        texture_size=args.texture_size,
        upscale=args.upscale,
        enhance_texture_angles=args.enhance_texture_angles
    )
    t5 = time.time()
    print(f"Texture generation took {t5 - t4:.2f} seconds")

    os.makedirs(args.output_dir, exist_ok=True)

    # Use mesh file name as output name
    output_name = os.path.splitext(os.path.basename(args.mesh_path))[0] + '_textured'

    mesh.export(os.path.join(args.output_dir, '{}.glb'.format(output_name)))

    print(f"Output saved to {args.output_dir}/{output_name}.glb")
    print(f"Total time taken: {t5 - t0:.2f} seconds")


if __name__ == "__main__":
    # Parse arguments and then call run
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_paths', type=str, nargs='+', default=None,
                        help='Path to input images. Can specify multiple paths separated by spaces')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt for the image')
    parser.add_argument('--mesh_path', type=str, help='Path to input mesh', required=True)
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('--texture_size', type=int, default=2048,
                        help='Resolution size of the texture used for the GLB')
    parser.add_argument('--upscale', action='store_true', help='Upscale the texture', default=False)
    parser.add_argument('--enhance_texture_angles', action='store_true', help='Enhance texture angles', default=False)
    parser.add_argument('--mmgp', action='store_true', default=False, help='Use MMGP offloading')
    parser.add_argument('--mmgp_profile', type=int, default=1)
    parser.add_argument('--mmgp_verbose', type=int, default=1)

    args = parser.parse_args()

    run(args)
