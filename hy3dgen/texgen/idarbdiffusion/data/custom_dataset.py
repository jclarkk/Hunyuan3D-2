import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


class CustomDataset(Dataset):
    def __init__(self,
        root_dir: str,
        object_list: list = None,
        ) -> None:
        self.root_dir = root_dir
        self.bg_color = np.array([1., 1., 1.], dtype=np.float32) # white

        if object_list is None:
            self.all_objects = [i for i in os.listdir(root_dir) if i.endswith('.png') ]
        else:
            self.all_objects = object_list

        print("loading ", len(self.all_objects), " objects in the dataset")

    def load_image(self, img_path, bg_color, alpha, return_type='np'):
        img = Image.open(img_path)

        # resize, max length, keep aspect ratio
        max_length = 512 # hard coded
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
        assert img.shape[-1] == 4, f"please input RGBA images, {img_path}"

        # pad to [max_length, max_length]
        h, w = img.shape[:2]
        pad_h, pad_w = (max_length - h)//2, (max_length - w)//2
        img = np.pad(img, ((pad_h, max_length - h - pad_h), (pad_w, max_length - w - pad_w), (0, 0)), mode='constant', constant_values=0)

        # crop image so that it can be diveded by 8
        # h, w = img.shape[:2]
        # nh, nw = h//8*8, w//8*8
        # dh, dw = int((h-nh)//2), int((w-nw)//2)
        # img = img[dh:dh+nh, dw:dw+nw]
        
        if img.shape[-1] == 3:
            alpha = np.ones_like(img[..., :1])
        else:
            alpha = (img[:, :, 3:] > 0.5).astype(np.float32)
        img = img[:, :, :3]

        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img, alpha

    def __len__(self):
        return len(self.all_objects)
    
    def __getitem__(self, index):
        data_path = self.all_objects[index]

        # get the bg color
        bg_color = self.bg_color

        img_path = os.path.join(self.root_dir, data_path)
        img_tensor, mask = self.load_image(img_path, bg_color, None, return_type='pt')
        img_tensors_in = img_tensor.permute(2, 0, 1).float()

        img_tensors_mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        task_ids = torch.tensor([0, 1, 2]).long()

        mv_pose = torch.tensor([torch.tensor(0), torch.tensor(0), (np.log(1.5) - np.log(1.2))/(np.log(2.2)-np.log(1.2)) * torch.pi, torch.tensor(0)])
        
        item = {
            'imgs_in': img_tensors_in[None,...], # (1, 3, H, W)
            # 'imgs_out': img_tensors_out, # (Nd, 3, H, W) (albedo, normal, mr)
            'imgs_mask': img_tensors_mask[None,...], # (1, 1, H, W)
            'task_ids': task_ids[None,...], # (ND, ND)
            'pose': mv_pose[None,...],
        }

        item['data_name'] = os.path.basename(data_path)[:-4] # filename
        
        return item
