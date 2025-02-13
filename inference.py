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

    # Generate mesh
    t2 = time.time()
    mesh = mesh_pipeline(image=image, num_inference_steps=30, mc_algo='mc',
                         generator=torch.manual_seed(args.seed))[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=args.face_count, im_remesh=args.im_remesh)
    t3 = time.time()
    print(f"Mesh generation took {t3 - t2:.2f} seconds")

    # Generate texture
    if args.texture:
        del mesh_pipeline
        torch.cuda.empty_cache()

        texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
        print('3D Paint pipeline loaded')

        t4 = time.time()
        mesh = texture_pipeline(mesh, image=image)
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
    parser.add_argument('--im_remesh', action='store_true', help='Remesh using InstantMeshes', default=False)
    parser.add_argument('--face_count', type=int, default=100000, help='Maximum face count for the mesh')
    parser.add_argument('--texture', action='store_true', help='Texture the mesh', default=False)

    args = parser.parse_args()

    run(args)
