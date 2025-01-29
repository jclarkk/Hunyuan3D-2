import numpy as np
import torch
import trimesh
from PIL import Image

from hy3dgen.texgen.differentiable_renderer.mesh_utils import save_mesh
from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
from hy3dgen.texgen.idarbdiffusion.pipeline import IDArbPipeline
from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap


def save_debug_image(tensor, filename):
    """Ensure tensor is in a valid shape before saving as image."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # If tensor has extra dimensions, remove them
    while tensor.ndim > 3:
        tensor = tensor.squeeze(0)

    if tensor.ndim == 2:  # Grayscale
        img = Image.fromarray((tensor * 255).astype(np.uint8))
    elif tensor.ndim == 3:
        if tensor.shape[-1] == 1:  # Single-channel, convert to grayscale
            tensor = tensor.squeeze(-1)
            img = Image.fromarray((tensor * 255).astype(np.uint8))
        elif tensor.shape[-1] == 3:  # RGB image
            img = Image.fromarray((tensor * 255).astype(np.uint8), "RGB")
        else:
            raise ValueError(f"Unexpected channel shape: {tensor.shape}")
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    img.save(filename)


def normalize_pbr_map(image):
    """Ensure that the image is properly formatted for baking."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().float()

    if isinstance(image, np.ndarray):
        image = torch.tensor(image, dtype=torch.float32)

    # Ensure image range is [0,1]
    if image.max() > 1:
        image /= 255.0

    # Ensure shape is (H, W, C)
    if image.ndim == 2:  # Convert grayscale to 3-channel RGB
        image = image.unsqueeze(-1).repeat(1, 1, 3)

    return image

def convert_to_pil(image):
    """Convert a NumPy array or Tensor to a PIL Image."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if isinstance(image, np.ndarray):
        # Ensure values are in the 0-255 range
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)

        # Convert grayscale to RGB if necessary
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        return Image.fromarray(image)

    elif isinstance(image, Image.Image):
        return image  # Already a PIL image

    raise ValueError("Unsupported image format for conversion")

class Hunyuan3DTexGenConfig:

    def __init__(self):
        self.device = 'cuda'

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.1, 0.1]

        self.candidate_camera_azims_enhanced = [0, 90, 180, 270, 0, 180, 90, 270, 45, 135, 225, 310, 45, 135, 225, 310]
        self.candidate_camera_elevs_enhanced = [0, 0, 0, 0, 90, -90, -45, -45, 15, 15, 15, 15, -15, -15, -15, -15]
        self.candidate_view_weights_enhanced = [1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                                0.1]

        self.render_size = 2048
        self.texture_size = 1024
        self.bake_exp = 4
        self.merge_method = 'fast'


config = Hunyuan3DTexGenConfig()

render = MeshRender(
    default_resolution=config.render_size,
    texture_size=config.texture_size
)

mesh = trimesh.load_mesh('/mnt/c/Users/jonathan/Downloads/60ab63ec6b3e4fc4ace4c835e4e93094.glb')

mesh = mesh_uv_wrap(mesh)

render.load_mesh(mesh)

def bake_from_multiview(views, camera_elevs,
                        camera_azims, view_weights, method='fast'):
    project_textures, project_weighted_cos_maps = [], []
    project_boundary_maps = []
    for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights):
        project_texture, project_cos_map, project_boundary_map = render.back_project(
            view, camera_elev, camera_azim)
        project_cos_map = weight * (project_cos_map ** config.bake_exp)
        project_textures.append(project_texture)
        project_weighted_cos_maps.append(project_cos_map)
        project_boundary_maps.append(project_boundary_map)

    for i, proj_texture in enumerate(project_textures):
        save_debug_image(proj_texture, f"debug_projection_{i}.png")

    if method == 'fast':
        texture, ori_trust_map = render.fast_bake_texture(
            project_textures, project_weighted_cos_maps)

        texture = texture.clone().detach().float()
        texture = render.color_rgb_to_srgb(texture)
    else:
        raise f'no method {method}'
    return texture, ori_trust_map > 1E-8


def texture_inpaint(texture, mask):
    texture_np = render.uv_inpaint(texture, mask)
    texture = torch.tensor(texture_np / 255).float().to(texture.device)
    return texture


normal_map = None
metalness_roughness_map = None

multiviews = [
    Image.open('multiview_0.png'),
    Image.open('multiview_1.png'),
    Image.open('multiview_2.png'),
    Image.open('multiview_3.png'),
    Image.open('multiview_4.png'),
    Image.open('multiview_5.png'),
]

selected_camera_elevs, selected_camera_azims, selected_view_weights = \
                (config.candidate_camera_elevs, config.candidate_camera_azims,
                 config.candidate_view_weights)

# Ensure multiviews are RGBA
multiviews = [img.convert("RGBA") for img in multiviews]

print('Creating PBR maps...')
pbr_pipeline = IDArbPipeline.from_pretrained('cuda')
pbr_images = pbr_pipeline(multiviews)

# Save maps for debugging
for i, pbr_image_maps in enumerate(pbr_images):
    albedo = pbr_image_maps["albedo"]
    normal = pbr_image_maps["normal"]
    roughness = pbr_image_maps["roughness"]
    metalness = pbr_image_maps["metalness"]

    print(f"🔍 Saving PBR maps for view {i} - Shapes: "
          f"Albedo={albedo.shape}, Normal={normal.shape}, Roughness={roughness.shape}, Metalness={metalness.shape}")

    Image.fromarray(albedo).save(f"debug_albedo_{i}.png")
    Image.fromarray(normal).save(f"debug_normal_{i}.png")
    Image.fromarray(roughness).save(f"debug_roughness_{i}.png")
    Image.fromarray(metalness).save(f"debug_metalness_{i}.png")

diffusion_maps = []
normal_maps = []
roughness_maps = []
metalness_maps = []

for pbr_image_maps in pbr_images:
    diffusion_maps.append(pbr_image_maps['albedo'])
    normal_maps.append(pbr_image_maps['normal'])
    roughness_maps.append(pbr_image_maps['roughness'])
    metalness_maps.append(pbr_image_maps['metalness'])

diffusion_maps = [normalize_pbr_map(img) for img in diffusion_maps]
normal_maps = [normalize_pbr_map(img) for img in normal_maps]
roughness_maps = [normalize_pbr_map(img) for img in roughness_maps]
metalness_maps = [normalize_pbr_map(img) for img in metalness_maps]

print('Baking PBR textures...')
texture, mask = bake_from_multiview(diffusion_maps,
                                    selected_camera_elevs, selected_camera_azims,
                                    selected_view_weights,
                                    method=config.merge_method)
# Debug texture map
Image.fromarray((texture.detach().cpu().numpy() * 255).astype(np.uint8)).save('debug_albedo_texture.png')

normal_map, normal_mask = bake_from_multiview(
    normal_maps,
    selected_camera_elevs,
    selected_camera_azims,
    selected_view_weights,
    method=config.merge_method
)
roughness_map, roughness_mask = bake_from_multiview(
    roughness_maps,
    selected_camera_elevs,
    selected_camera_azims,
    selected_view_weights,
    method=config.merge_method
)
metalness_map, metalness_mask = bake_from_multiview(
    metalness_maps,
    selected_camera_elevs,
    selected_camera_azims,
    selected_view_weights,
    method=config.merge_method
)

metalness_roughness_map = pbr_pipeline.combine_roughness_metalness(
    np.asarray(roughness_map.detach().cpu()),
    np.asarray(metalness_map.detach().cpu())
)

mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

print('Inpainting texture...')
texture = texture_inpaint(texture, mask_np)

texture = texture.cpu()
normal_map = normal_map.cpu()

albedo_texture = convert_to_pil(texture)
normal_texture = convert_to_pil(normal_map)
metalness_roughness_texture = convert_to_pil(metalness_roughness_map)

# Save images for debugging
albedo_texture.save('debug_albedo_texture_inpaint.png')
normal_texture.save('debug_normal_texture_inpaint.png')
metalness_roughness_texture.save('debug_metalness_roughness_texture_inpaint.png')

render.set_texture(texture)
textured_mesh = render.save_mesh(normal_texture, metalness_roughness_texture)
textured_mesh.export('pbr.glb')
