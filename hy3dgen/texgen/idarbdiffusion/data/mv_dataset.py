import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import json
import gzip
import numpy as np
from PIL import Image
import random
from typing import Literal, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset


def srgb_to_rgb_blender(f: np.ndarray) -> np.ndarray:
    # https://docs.blender.org/manual/en/latest/glossary/index.html#term-Color-Space
    # gamma 2.2
    f = np.power(f, 2.2)
    return f


def get_mv_info(data_type, abs_info, root_dir, views=3, split='train'):
    info = []
    if data_type == 'gobjaverse':
        # root_dir = '__gobjaverse__' # TODO: root dir of GObjaverse
        obj_dir = abs_info
        if split == 'test':
            assert views == 4
            view_list = [0, 6, 12, 18]
        else:
            view_list = list(range(0, 24)) + list(range(25, 39))
            view_list = random.sample(view_list, views)
        for view in view_list:
            info.append(
                {
                    'mask': None,
                    'image': os.path.join(root_dir, obj_dir, f'{view:05d}', f'{view:05d}.png'),
                    'albedo': os.path.join(root_dir, obj_dir, f'{view:05d}', f'{view:05d}_albedo.png'),
                    'normal': os.path.join(root_dir, obj_dir, f'{view:05d}', f'{view:05d}_nd.exr'),
                    'camera': os.path.join(root_dir, obj_dir, f'{view:05d}', f'{view:05d}.json'),
                    'material': os.path.join(root_dir, obj_dir, f'{view:05d}', f'{view:05d}_mr.png'),
                }
            )
    elif data_type == 'abo':
        # root_dir = '__abo__' # TODO: root dir of ABO
        obj_dir, scenes = abs_info.split('/')
        scenes = scenes.split('_')
        view_list = list(range(0, 91 * len(scenes)))
        view_list = random.sample(view_list, views)
        for view in view_list:
            scene = scenes[view // 91]
            view_idx = view % 91
            info.append(
                {
                    'mask': os.path.join(root_dir, obj_dir, 'segmentation', f'segmentation_{view_idx}.jpg'),
                    'image': os.path.join(root_dir, obj_dir, 'render', scene, f'render_{view_idx}.jpg'),
                    'albedo': os.path.join(root_dir, obj_dir, 'base_color', f'base_color_{view_idx}.jpg'),
                    'normal': os.path.join(root_dir, obj_dir, 'normal', f'normal_{view_idx}.png'),
                    'camera': (os.path.join(root_dir, obj_dir, 'metadata.json'), view_idx),
                    'material': os.path.join(root_dir, obj_dir, 'metallic_roughness', f'metallic_roughness_{view_idx}.jpg'),
                }
            )
    elif data_type == 'arbobjaverse':
        # root_dir = '__arbobjaverse__' # TODO: root dir of ArbObjaverse
        obj_dir = abs_info
        if split == 'test':
            scene_views = [(0, 0), (1, 3), (2, 6), (3, 9)]
        else:
            scene_views = [(i,j) for i in range(7) for j in range(12)]
            scene_views = random.sample(scene_views, views)
        for scene, view in scene_views:
            info.append(
                {
                    'mask': None,
                    'image': os.path.join(root_dir, obj_dir, f'color_{scene}_{view:02d}.png'),
                    'albedo': os.path.join(root_dir, obj_dir, f'albedo_{view:02d}.png'),
                    'normal': os.path.join(root_dir, obj_dir, f'normal_{view:02d}.exr'),
                    'camera': (os.path.join(root_dir, obj_dir, f'camera.json'), view),
                    'material': os.path.join(root_dir, obj_dir, f'orm_{view:02d}.png'),
                }
            )
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    return info


class MVDataset(Dataset):
    def __init__(self,
        img_wh: Tuple[int, int],
        object_list: str,
        data_root: dict,
        split: str = 'train',
        num_validation_samples: int = None,
        num_samples: Optional[int] = None,
        num_views: int = 3,
        ) -> None:

        self.num_samples = num_samples
        self.img_wh = img_wh
        self.num_views = num_views
        self.split = split
        self.data_root = data_root
        self.bg_color = np.array([1., 1., 1.], dtype=np.float32) # white

        if object_list.endswith('.gz'):
            self.all_objects = list(json.load(gzip.open(object_list, 'rt')))
        else:
            self.all_objects = list(json.load(open(object_list)))

        if num_validation_samples is not None:
            if split == 'train':
                self.all_objects = self.all_objects[:-num_validation_samples]
            elif split == 'val':
                self.all_objects = self.all_objects[-num_validation_samples:]

        if num_samples is not None:
            self.all_objects = self.all_objects[:num_samples]

        print("loading ", len(self.all_objects), " objects in the dataset")

    def __len__(self):
        return len(self.all_objects)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def load_mask(self, img_path, return_type='np'):
        if img_path is None:
            return None

        img = Image.open(img_path)
        img = np.array(img.resize(self.img_wh))
        img = (img > 128).astype(np.float32)

        if len(np.shape(img)) == 3:
            img = img.mean(axis=-1, keepdims=True)
        elif len(np.shape(img)) == 2:
            img = img[..., None]
        else:
            raise ValueError

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError
        
        return img

    def load_image(self, img_path, bg_color, alpha, return_type='np', do_gamma=False):
        img = Image.open(img_path)
        img = np.array(img.resize(self.img_wh))
        img = img.astype(np.float32) / 255. # [0, 1]
        assert img.shape[-1] == 3 or img.shape[-1] == 4 # RGB or RGBA

        if alpha is None and img.shape[-1] == 4:
            alpha = img[:, :, 3:]
            img = img[:, :, :3]

        if do_gamma:
            img = img ** (1/2.2)

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img, alpha

    def load_normal_gob(self, normal_path, cond_c2w, bg_color, alpha, return_type='np'):
        """
        Load normal map and transform it to normal bae system
        Adpated from https://github.com/modelscope/richdreamer/blob/main/dataset/gobjaverse/process_unity_dataset.py
        """
        def unity2blender(normal):
            normal_clone = normal.copy()
            normal_clone[...,0] = -normal[...,-1]
            normal_clone[...,1] = -normal[...,0]
            normal_clone[...,2] = normal[...,1]

            return normal_clone

        def blender2midas(img):
            '''Blender: rub
            midas: lub
            '''
            img[...,0] = -img[...,0]
            img[...,1] = -img[...,1]
            img[...,-1] = -img[...,-1]
            return img

        normald = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
        normald = normald.astype(np.float32)
        normald = cv2.resize(normald, self.img_wh, interpolation=cv2.INTER_CUBIC)
        normal = normald[...,:3]
        normal_norm = (np.linalg.norm(normal, 2, axis=-1, keepdims= True))
        normal = normal / normal_norm
        normal = np.nan_to_num(normal,nan=-1.)

        world_normal = unity2blender(normal)

        img = blender2midas(world_normal@ (cond_c2w[:3,:3]))
        img = np.clip((img+1.)/2., 0., 1.)

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def load_normal_abo(self, normal_path, bg_color, alpha, return_type='np'):
        """
        Load normal map and transform it to normal bae system
        Adpated from https://github.com/modelscope/richdreamer/blob/main/dataset/gobjaverse/process_unity_dataset.py
        """

        def blender2midas(img):
            '''Blender: rub
            midas: lub
            '''
            img[...,0] = -img[...,0]
            return img

        normald = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
        normald = normald.astype(np.float32)
        normald = cv2.resize(normald, self.img_wh, interpolation=cv2.INTER_CUBIC)
        normal = normald[...,::-1] / 255. * 2. - 1.

        normal_norm = (np.linalg.norm(normal, 2, axis=-1, keepdims= True))
        normal = normal / (normal_norm + 1e-6)
        normal = np.nan_to_num(normal,nan=-1.)

        img = blender2midas(normal)
        img = np.clip((img+1.)/2., 0., 1.)

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def load_normal_arbo(self, normal_path, bg_color, alpha, return_type='np'):
        def blender2midas(img_o):
            '''Blender: rub
            midas: lub
            '''
            img = img_o.copy()
            img[..., 0] = -img_o[..., -1]
            img[..., -1] = img_o[..., 0]
            return img
        
        normald = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
        normald = normald.astype(np.float32)
        normald = cv2.resize(normald, self.img_wh, interpolation=cv2.INTER_CUBIC)
        normal = normald[...,::-1] * 2. - 1.

        normal_norm = (np.linalg.norm(normal, 2, axis=-1, keepdims= True))
        normal = normal / (normal_norm + 1e-6)
        normal = np.nan_to_num(normal,nan=-1.)

        img = blender2midas(normal)
        img = np.clip((img+1.)/2., 0., 1.)

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def load_mr_gob(self, img_path, bg_color, alpha, return_type='np'):
        img = Image.open(img_path)
        img = np.array(img.resize(self.img_wh))
        img = img.astype(np.float32) / 255. # [0, 1]
        img = img[...,:3]

        img = srgb_to_rgb_blender(img) # Note!!! material.png is in sRGB

        out_imgs = []
        out_imgs.append(img)

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]
        out_imgs = [img[...,:3] * alpha + bg_color * (1 - alpha) for img in out_imgs]

        if return_type == "np":
            pass
        elif return_type == "pt":
            out_imgs = [torch.from_numpy(img) for img in out_imgs]
        else:
            raise NotImplementedError

        return out_imgs

    def load_mr_abo(self, img_path, bg_color, alpha, return_type='np'):
        img = Image.open(img_path)
        img = np.array(img.resize(self.img_wh))
        img = img.astype(np.float32) / 255. # [0, 1]

        # orm -> mro
        img[..., 0] = img[..., -1]
        img[..., -1] = 0
        out_imgs.append(img)

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]
        out_imgs = [img[...,:3] * alpha + bg_color * (1 - alpha) for img in out_imgs]

        if return_type == "np":
            pass
        elif return_type == "pt":
            out_imgs = [torch.from_numpy(img) for img in out_imgs]
        else:
            raise NotImplementedError

        return out_imgs

    def load_c2w_gob(self, camera_json):
        json_content = json.load(open(camera_json))

        # suppose is true
        camera_matrix = np.eye(4)
        camera_matrix[:3, 0] = np.array(json_content['x'])
        camera_matrix[:3, 1] = np.array(json_content['y'])
        camera_matrix[:3, 2] = np.array(json_content['z'])
        camera_matrix[:3, 3] = np.array(json_content['origin'])

        return camera_matrix
    
    def get_pose(self, c2w):
        origin = c2w[:3, -1]
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(origin[None, :])
        if z_target.item() < 1.2 or z_target.item() > 2.2:
            z_target = np.clip(z_target.item(), 1.2, 2.2)
        target_T = torch.tensor([theta_target.item(), azimuth_target.item(), (np.log(z_target.item()) - np.log(1.2))/(np.log(2.2)-np.log(1.2)) * torch.pi, torch.tensor(0)])
        assert torch.all(target_T <= torch.pi) and torch.all(target_T >= -torch.pi)
        return target_T

    def load_c2w_abo(self, camera_info):
        camera_json, view_idx = camera_info
        json_content = json.load(open(camera_json))
        camera_matrix = np.array(json_content['views'][view_idx]['pose']).reshape(4, 4)
        return camera_matrix
    
    def load_c2w_arbo(self, camera_info):
        camera_json, view_idx = camera_info
        json_content = json.load(open(camera_json))
        camera_matrix = np.array(json_content['transform_matrix'][view_idx]).reshape(4, 4)
        return camera_matrix
    
    def __len__(self):
        return len(self.all_objects)

    def __fetch_data(self, data_type, object_info):

        # get the bg color
        bg_color = self.bg_color

        # load the mask
        mask = self.load_mask(object_info['mask'], return_type='np')
        # load the image
        img_tensors_in, mask = self.load_image(object_info['image'], bg_color, mask, return_type='pt')
        img_tensors_in = img_tensors_in.permute(2, 0, 1).float()

        img_tensors_out = []
        # load albedo
        do_gamma = False
        img_tensor, _ = self.load_image(object_info['albedo'], bg_color, mask, return_type='pt', do_gamma=do_gamma)
        img_tensors_out.append(img_tensor.permute(2, 0, 1))

        # load camera pose
        if data_type == 'gobjaverse':
            camera2world = self.load_c2w_gob(object_info['camera'])
        elif data_type == 'abo': # ABO
            camera2world = self.load_c2w_abo(object_info['camera'])
        else: # arbobjaverse
            camera2world = self.load_c2w_arbo(object_info['camera'])
        pose = self.get_pose(camera2world)

        # load normal
        if data_type == 'gobjaverse':
            normal_tensor = self.load_normal_gob(object_info['normal'], camera2world, bg_color, mask, return_type='pt')
        elif data_type == 'abo':
            normal_tensor = self.load_normal_abo(object_info['normal'], bg_color, mask, return_type='pt')
        elif data_type == 'arbobjaverse':
            normal_tensor = self.load_normal_arbo(object_info['normal'], bg_color, mask, return_type='pt')
        else:
            raise NotImplementedError
        img_tensors_out.append(normal_tensor.permute(2, 0, 1))

        # load metrial
        if data_type == 'gobjaverse':
            mr_tensors = self.load_mr_gob(object_info['material'], bg_color, mask, return_type='pt')
        elif data_type == 'abo' or data_type == 'arbobjaverse':
            mr_tensors = self.load_mr_abo(object_info['material'], bg_color, mask, return_type='pt')
        else:
            raise NotImplementedError
        img_tensors_out.extend([mr_tensor.permute(2, 0, 1) for mr_tensor in mr_tensors])
        
        img_tensors_out = torch.stack(img_tensors_out, dim=0).float()
        img_tensors_mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        task_inds = torch.tensor([0, 1, 2]).long()

        return img_tensors_in, img_tensors_out, img_tensors_mask, task_inds, pose
    
    def __fetch_mv(self, index):
        data_type, data_info = self.all_objects[index]
        data_root = self.data_root[data_type]
        object_infos = get_mv_info(data_type, data_info, data_root, views=self.num_views, split=self.split)

        mv_img_in, mv_img_out, mv_img_mask, mv_task_ids, mv_pose = [], [], [], [], []
        for info in object_infos:
            img_in, img_out, img_mask, task_ids, pose = self.__fetch_data(data_type, info)
            mv_img_in.append(img_in)
            mv_img_out.append(img_out)
            mv_img_mask.append(img_mask)
            mv_task_ids.append(task_ids)
            mv_pose.append(pose)
        
        mv_img_in = torch.stack(mv_img_in, dim=0)
        mv_img_out = torch.stack(mv_img_out, dim=0)
        mv_img_mask = torch.stack(mv_img_mask, dim=0)
        mv_task_ids = torch.stack(mv_task_ids, dim=0)
        mv_pose = torch.stack(mv_pose, dim=0)

        item = {
            'imgs_in': mv_img_in, # (Nv, 3, H, W)
            'imgs_out': mv_img_out, # (Nv, Nd, 3, H, W) (albedo, normal, mr)
            'imgs_mask': mv_img_mask, # (Nv, 1, H, W)
            'task_ids': mv_task_ids, # (Nv, ND)
            'pose': mv_pose, # (Nv, 4)
        }

        return item
    
    def __getitem__(self, index):
        try:
            item = self.__fetch_mv(index)
        except:
            print("Error loading data", self.all_objects[index])
            item = self.__fetch_mv(0)
        return item


if __name__ == "__main__":
    pass
