import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class RMBGRemover:
    def __call__(self, input: Image.Image, height=518, width=518,
                 background_color: list[float] = [1.0, 1.0, 1.0]) -> Image.Image:
        """
        Preprocess the input image with a customizable background color.
        background_color: List of 3 floats [R, G, B] in range 0-1 (default: [1.0, 1.0, 1.0] for white)
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
            del model
            torch.cuda.empty_cache()
        output = input
        # Crop and resize based on alpha channel after background removal
        output_np = np.array(output)
        # Ensure the image is in RGBA mode before accessing the alpha channel
        if output.mode != 'RGBA':
            output = output.convert('RGBA')
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((height, width), Image.Resampling.LANCZOS)

        bg_rgb = tuple(int(c * 255) for c in background_color)
        colored_bg = Image.new('RGBA', output.size, bg_rgb + (255,))
        # Ensure output has alpha channel
        if output.mode != 'RGBA':
            output = output.convert('RGBA')
        # Composite output onto colored background preserving alpha
        colored_bg.alpha_composite(output)

        return colored_bg
