import cvcuda
import torch
from typing import Optional

torch_to_cvc = lambda x, layout: cvcuda.as_tensor(x, layout)

cvc_to_torch = lambda x, device: torch.tensor(x.cuda(), device=device)


def inpaint_cvc(
    image: torch.Tensor,
    mask: torch.Tensor,
    padding_size: int,
    return_dtype: Optional[torch.dtype] = None,
):
    input_dtype = image.dtype
    input_device = image.device

    image = image.detach()
    mask = mask.detach()

    if image.dtype != torch.uint8:
        image = (image * 255).to(torch.uint8)
    if mask.dtype != torch.uint8:
        mask = (mask * 255).to(torch.uint8)

    image_cvc = torch_to_cvc(image, "HWC")
    mask_cvc = torch_to_cvc(mask, "HW")
    output_cvc = cvcuda.inpaint(image_cvc, mask_cvc, padding_size)
    output = cvc_to_torch(output_cvc, device=input_device)

    if return_dtype == torch.uint8 or input_dtype == torch.uint8:
        return output
    return output.to(dtype=input_dtype) / 255.0


def batch_inpaint_cvc(
    images: torch.Tensor,
    masks: torch.Tensor,
    padding_size: int,
    return_dtype: Optional[torch.dtype] = None,
):
    output = torch.stack(
        [
            inpaint_cvc(image, mask, padding_size, return_dtype)
            for (image, mask) in zip(images, masks)
        ],
        axis=0,
    )
    return output


def batch_erode(
    masks: torch.Tensor, kernel_size: int, return_dtype: Optional[torch.dtype] = None
):
    input_dtype = masks.dtype
    input_device = masks.device
    masks = masks.detach()
    if masks.dtype != torch.uint8:
        masks = (masks.float() * 255).to(torch.uint8)
    masks_cvc = torch_to_cvc(masks[..., None], "NHWC")
    masks_erode_cvc = cvcuda.morphology(
        masks_cvc,
        cvcuda.MorphologyType.ERODE,
        maskSize=(kernel_size, kernel_size),
        anchor=(-1, -1),
    )
    masks_erode = cvc_to_torch(masks_erode_cvc, device=input_device)[..., 0]
    if return_dtype == torch.uint8 or input_dtype == torch.uint8:
        return masks_erode
    return (masks_erode > 0).to(dtype=input_dtype)


def batch_dilate(
    masks: torch.Tensor, kernel_size: int, return_dtype: Optional[torch.dtype] = None
):
    input_dtype = masks.dtype
    input_device = masks.device
    masks = masks.detach()
    if masks.dtype != torch.uint8:
        masks = (masks.float() * 255).to(torch.uint8)
    masks_cvc = torch_to_cvc(masks[..., None], "NHWC")
    masks_dilate_cvc = cvcuda.morphology(
        masks_cvc,
        cvcuda.MorphologyType.DILATE,
        maskSize=(kernel_size, kernel_size),
        anchor=(-1, -1),
    )
    masks_dilate = cvc_to_torch(masks_dilate_cvc, device=input_device)[..., 0]
    if return_dtype == torch.uint8 or input_dtype == torch.uint8:
        return masks_dilate
    return (masks_dilate > 0).to(dtype=input_dtype)
