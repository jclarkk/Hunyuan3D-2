import argparse
import os
import time
import torch
from PIL import Image
from mmgp import offload
from uuid import uuid4

from hy3dgen.rmbg import preprocess_image
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
    image = preprocess_image(image)

    processed_image_name = os.path.basename(image_path).split('.')[0] + '_input.png'
    image.save(os.path.join(args.output_dir, processed_image_name))

    t1 = time.time()
    print(f"Image processing took {t1 - t0:.2f} seconds")

    if args.mmgp:
        mesh_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2', device="cpu",
                                                                         use_safetensors=True, use_mmgp=True)
    else:
        mesh_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2', use_safetensors=True)
    print('3D DiT pipeline loaded')

    if args.texture:
        texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2', use_mmgp=args.mmgp)
        print('3D Paint pipeline loaded')
    else:
        texture_pipeline = None

    # Handle MMGP offloading
    if args.mmgp:
        profile = args.mmgp_profile
        kwargs = {}

        pipe = offload.extract_models("i23d_worker", mesh_pipeline)

        if args.texture:
            pipe.update(offload.extract_models("texgen_worker", texture_pipeline))
            texture_pipeline.models["multiview_model"].pipeline.vae.use_slicing = True

        if profile < 5:
            kwargs["pinnedMemory"] = "i23d_worker/model"
        if profile != 1 and profile != 3:
            kwargs["budgets"] = {"*": 2200}

        offload.profile(pipe, profile_no=profile, verboseLevel=args.mmgp_verbose, **kwargs)

    # Generate mesh
    t2 = time.time()
    mesh = mesh_pipeline(image=image, num_inference_steps=30, mc_algo='mc',
                         generator=torch.manual_seed(args.seed), use_mmgp=args.mmgp)[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=args.face_count, im_remesh=args.im_remesh)
    t3 = time.time()
    print(f"Mesh generation took {t3 - t2:.2f} seconds")

    # Generate texture
    if args.texture:
        t4 = time.time()
        mesh = texture_pipeline(mesh, image=image_path, texture_size=args.texture_size)
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


if __name__ == "__main__":
    # Parse arguments and then call run
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_paths', type=str, nargs='+', required=True,
                        help='Path to input images. Can specify multiple paths separated by spaces')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('--texture_size', type=int, default=2048,
                        help='Resolution size of the texture used for the GLB')
    parser.add_argument('--im_remesh', action='store_true', help='Remesh using InstantMeshes', default=False)
    parser.add_argument('--face_count', type=int, default=100000, help='Maximum face count for the mesh')
    parser.add_argument('--texture', action='store_true', help='Texture the mesh', default=False)
    parser.add_argument('--mmgp', action='store_true', default=False, help='Use MMGP offloading')
    parser.add_argument('--mmgp_profile', type=int, default=1)
    parser.add_argument('--mmgp_verbose', type=int, default=1)

    args = parser.parse_args()

    run(args)
