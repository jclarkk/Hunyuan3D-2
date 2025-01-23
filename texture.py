import argparse
import os
import time
import torch.cuda
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

    if args.texture_size not in [2048, 4096]:
        raise ValueError("Texture size must either be 2k or 4k")

    if args.prompt is not None:
        t0 = time.time()
        t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        image = t2i(args.prompt)
        t1 = time.time()
        print(f"Text to image took {t1 - t0:.2f} seconds")
        del t2i
        torch.cuda.empty_cache()
    else:
        # Only one image supported right now
        image_path = args.image_paths[0]
        image = Image.open(image_path)

    t2 = time.time()

    # Preprocess the image
    image = preprocess_image(image)

    t3 = time.time()
    print(f"Image processing took {t3 - t2:.2f} seconds")

    # Load mesh
    mesh = trimesh.load_mesh(args.mesh_path)

    # Reduce face count
    ms = import_mesh(mesh)
    ms = reduce_face(ms, max_facenum=50000)
    current_mesh = ms.current_mesh()
    mesh = trimesh.Trimesh(vertices=current_mesh.vertex_matrix(), faces=current_mesh.face_matrix())

    # Generate texture
    t4 = time.time()
    pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
    mesh = pipeline(mesh, image=image, texture_size=args.texture_size, upscale=args.upscale)
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

    args = parser.parse_args()

    run(args)
