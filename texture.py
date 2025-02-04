import argparse
import os
import time
import trimesh
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

    t2i_pipeline = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
    texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')

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
        ms = reduce_face(ms, max_facenum=90000)
        current_mesh = ms.current_mesh()
        mesh = trimesh.Trimesh(vertices=current_mesh.vertex_matrix(), faces=current_mesh.face_matrix())

    # Generate texture
    t4 = time.time()
    mesh = texture_pipeline(
        mesh,
        image=image,
        upscale_model=args.upscale_model,
        enhance_texture_angles=args.enhance_texture_angles,
        pbr=args.pbr,
        debug=args.debug
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
    parser.add_argument('--upscale_model', type=str, default=None, help='Upscale model to use')
    parser.add_argument('--enhance_texture_angles', action='store_true', help='Enhance texture angles', default=False)
    parser.add_argument('--pbr', action='store_true', help='Generate PBR textures', default=False)
    parser.add_argument('--debug', action='store_true', help='Debug mode', default=False)

    args = parser.parse_args()

    run(args)
