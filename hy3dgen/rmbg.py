import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class RMBGRemover:

    def __init__(self, local_files_only=False):
        self.local_files_only = local_files_only

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
            model = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet_HR',
                                                                  trust_remote_code=True,
                                                                  local_files_only=self.local_files_only)
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
        W, H = output.size
        alpha = np.array(output)[:, :, 3]

        ys, xs = np.where(alpha > 0.8 * 255)
        if xs.size == 0 or ys.size == 0:
            return output.convert('RGB')

        pad_px = 8
        x0 = max(xs.min() - pad_px, 0)
        y0 = max(ys.min() - pad_px, 0)
        x1 = min(xs.max() + pad_px, W - 1)
        y1 = min(ys.max() + pad_px, H - 1)

        cropped = output.crop((x0, y0, x1, y1))

        short_side = max(cropped.size)
        square = Image.new('RGBA', (short_side, short_side), (0, 0, 0, 0))
        off_x = (short_side - cropped.width)  // 2
        off_y = (short_side - cropped.height) // 2
        square.alpha_composite(cropped, (off_x, off_y))

        target = 1024
        scale = target / short_side
        new_w  = int(round(cropped.width  * scale))
        new_h  = int(round(cropped.height * scale))

        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

        canvas = Image.new('RGBA', (target, target), (0, 0, 0, 0))
        canvas.alpha_composite(resized,
                               ((target - new_w) // 2, (target - new_h) // 2))

        bg_rgb = tuple(int(c * 255) for c in background_color)
        bg = Image.new('RGBA', canvas.size, bg_rgb + (255,))
        bg.alpha_composite(canvas)

        return bg
