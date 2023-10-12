import os
import pandas as pd
from torchvision.io import read_image, ImageReadMode
from torchvision.datasets.mnist import FashionMNIST
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import PIL

import dataloader.transform

__all__ = ['get_dataloader']

def get_dataloader(cfg):
    dataset = DiffusionImageDataset(cfg)
    return DataLoader(dataset, batch_size=cfg.loader.batch_size, 
                      shuffle=cfg.loader.shuffle, num_workers=cfg.loader.num_workers)

class DiffusionImageDataset(Dataset):
    def __init__(self, cfg):
        self.data_path = Path(cfg.loader.data_path)
        self.img_paths = list(self.data_path.glob("*.png"))
        self.img_shape = tuple(cfg.loader.img_shape)
        self.transform = getattr(dataloader.transform, cfg.loader.transform)()
        self.n_steps = cfg.model.params.n_steps
        for i in range(cfg.loader.extend_dataset_times):
            self.img_paths.extend(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = PIL.Image.open(str(self.img_paths[idx]) ).convert('RGB')
        image = image.resize(self.img_shape)
        t = torch.randint(0, self.n_steps, (1,))
        if self.transform:
            image = self.transform(image)

        target = torch.randn_like(image)
        return [image, t, target], target