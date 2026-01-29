import os 
from PIL import Image 
import torch 
from torch.utils.data import Dataset

class CustomDataset_1(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img
        
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, add_noise=None, K=1000):
        self.root_dir = root_dir
        self.transform = transform
        self.add_noise = add_noise
        self.K = K

        self.images = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)  # (C,H,W)

        imgs = img.unsqueeze(0).repeat(self.K, 1, 1, 1)
        noise = torch.randn_like(imgs)

        t_int = torch.randint(0, 10000, (self.K, 1, 1, 1))
        t = t_int.float() / 10000.0

        xt = self.add_noise(imgs, t, noise)

        return xt, t, noise, t_int


class CustomNoiseImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.K = dataset.K

    def __len__(self):
        return len(self.dataset) * self.K

    def __getitem__(self, idx):
        img_idx = idx // self.K
        t_idx = idx % self.K

        xt, t, noise, t_int = self.dataset[img_idx]

        return (
            xt[t_idx],
            t[t_idx],
            noise[t_idx],
            t_int[t_idx]
        )
