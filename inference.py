import argparse
import numpy as np
import os
import time
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from uuid import uuid4

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline


def preprocess_image(input: Image.Image) -> Image.Image:
    """
    Preprocess the input image with a white background.
    """
    # Check if the image has an alpha channel and whether to use it directly
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    if not has_alpha:
        model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        model.to('cuda')
        model.eval()
        # Data settings
        image_size = (1024, 1024)
        # Convert to RGB before applying transforms
        transform_image = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # Ensure image is in RGB mode for prediction
        input_rgb = input.convert('RGB')
        input_images = transform_image(input_rgb).unsqueeze(0).to('cuda')
        # Prediction
        with torch.no_grad():
            preds = model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(input.size)
        input.putalpha(mask)
        torch.cuda.empty_cache()
        del model
    output = input
    # Crop and resize based on alpha channel after background removal
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore
    output = output.resize((518, 518), Image.Resampling.LANCZOS)

    # Create a white background
    white_bg = Image.new('RGB', output.size, (255, 255, 255))

    # Composite the image over the white background
    if output.mode == 'RGBA':
        white_bg.paste(output, mask=output.split()[3])  # Use alpha channel as mask
        output = white_bg
    else:
        # If somehow the image is not RGBA, convert it
        output = output.convert('RGBA')
        white_bg.paste(output, mask=output.split()[3])
        output = white_bg

    return output


def run(args):
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

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')

    # Generate mesh
    t2 = time.time()
    mesh = pipeline(image=image, num_inference_steps=30, mc_algo='mc',
                    generator=torch.manual_seed(args.seed))[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    t3 = time.time()
    print(f"Mesh generation took {t3 - t2:.2f} seconds")

    # Generate texture
    t4 = time.time()
    pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
    mesh = pipeline(mesh, image=image_path, texture_size=args.texture_size)
    t5 = time.time()
    print(f"Texture generation took {t5 - t4:.2f} seconds")

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

    args = parser.parse_args()

    run(args)
