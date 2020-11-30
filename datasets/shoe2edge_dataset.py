import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as tr_func
from PIL import Image
import pytorch_lightning as pl
from argparse import ArgumentParser


class Shoe2EdgeDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int = 32, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = Shoe2EdgeDataset(self.data_dir, train=True)
        self.val_dataset = Shoe2EdgeDataset(self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str, required=True)
        parser.add_argument('--batch_size', type=int, default=32)
        return parser


class Shoe2EdgeDataset(Dataset):

    def __init__(self, data_dir, train=False):
        mode = 'train' if train else 'val'
        data_dir = os.path.join(data_dir, mode)
        files = os.listdir(data_dir)
        self.data = [os.path.join(data_dir, f) for f in files if f[-4:] == '.jpg']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item]
        image = Image.open(image)

        image = tr_func.to_tensor(image)
        image = tr_func.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        gt, image = image.split(256, dim=2)
        gt = 1 - gt
        gt = gt.mean(dim=0, keepdims=True)

        return image, gt
