import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


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
