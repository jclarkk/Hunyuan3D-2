import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class CustomMVDataset(Dataset):
    def __init__(self,
                 images: list,  # List of PIL Image objects
                 transforms_list: list = None,
                 num_views: int = 4,
                 use_cam: bool = False,
                 ) -> None:
        self.images = images
        self.transforms_list = transforms_list
        self.num_views = num_views
        self.use_cam = use_cam
        self.bg_color = np.array([1., 1., 1.], dtype=np.float32)
        print("loading ", len(self.images), " images in the dataset")

    def process_image(self, img, bg_color, alpha, return_type='np'):
        max_length = 512
        h, w = img.size
        if h > w:
            new_h = max_length
            new_w = int(w / h * max_length)
        else:
            new_w = max_length
            new_h = int(h / w * max_length)
        img = img.resize((new_h, new_w), Image.BICUBIC)
        img = np.array(img)
        img = img.astype(np.float32) / 255.
        assert img.shape[-1] == 4  # RGBA

        h, w = img.shape[:2]
        pad_h, pad_w = (max_length - h) // 2, (max_length - w) // 2
        img = np.pad(img, ((pad_h, max_length - h - pad_h), (pad_w, max_length - w - pad_w), (0, 0)),
                     mode='constant', constant_values=0)

        alpha = (img[:, :, 3:] > 0.5).astype(np.float32)
        img = img[:, :, :3]
        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "pt":
            img = torch.from_numpy(img)
        elif return_type != "np":
            raise NotImplementedError
        return img, alpha

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_pose(self, c2w):
        origin = c2w[:3, -1]
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(origin[None, :])
        if z_target.item() < 1.2 or z_target.item() > 2.2:
            z_target = np.clip(z_target.item(), 1.2, 2.2)
        target_T = torch.tensor([
            theta_target.item(),
            azimuth_target.item(),
            (np.log(z_target.item()) - np.log(1.2)) / (np.log(2.2) - np.log(1.2)) * torch.pi,
            torch.tensor(0)
        ])
        assert torch.all(target_T <= torch.pi) and torch.all(target_T >= -torch.pi)
        return target_T

    def __len__(self):
        return len(self.images) // self.num_views

    def __fetch_data(self, img, transform=None):
        img_tensor, mask = self.process_image(img, self.bg_color, None, return_type='pt')
        img_tensors_in = img_tensor.permute(2, 0, 1).float()
        img_tensors_mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        task_ids = torch.tensor([0, 1, 2]).long()

        if transform is not None and self.use_cam:
            c2w = np.array(transform)
            pose = self.get_pose(c2w)
        else:
            pose = torch.tensor([
                torch.tensor(0),
                torch.tensor(0),
                (np.log(1.5) - np.log(1.2)) / (np.log(2.2) - np.log(1.2)) * torch.pi,
                torch.tensor(0)
            ])

        return img_tensors_in, img_tensors_mask, task_ids, pose

    def __getitem__(self, index):
        start_idx = index * self.num_views
        mv_img_in, mv_img_mask, mv_task_ids, mv_pose = [], [], [], []

        for i in range(self.num_views):
            img = self.images[start_idx + i]
            transform = self.transforms_list[start_idx + i] if self.transforms_list else None

            img_in, img_mask, task_ids, pose = self.__fetch_data(img, transform)
            mv_img_in.append(img_in)
            mv_img_mask.append(img_mask)
            mv_task_ids.append(task_ids)
            mv_pose.append(pose)

        mv_img_in = torch.stack(mv_img_in, dim=0)
        mv_img_mask = torch.stack(mv_img_mask, dim=0)
        mv_task_ids = torch.stack(mv_task_ids, dim=0)
        mv_pose = torch.stack(mv_pose, dim=0)

        item = {
            'imgs_in': mv_img_in,
            'imgs_mask': mv_img_mask,
            'task_ids': mv_task_ids,
            'pose': mv_pose,
            'data_name': f'group_{index}'
        }

        return item
