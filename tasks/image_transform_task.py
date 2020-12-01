from typing import Union, Callable
import torch
from torch import nn
import pytorch_lightning as pl
from argparse import ArgumentParser


class ImageTransformTask(pl.LightningModule):

    def __init__(self, transformer: nn.Module, lr: float = 1e-3, loss_func: Union[str, nn.Module, Callable] = 'MAE', **kwargs):
        super().__init__()
        self.transformer = transformer
        self.lr = lr

        if isinstance(loss_func, str):
            if loss_func == 'MAE':
                self.loss_func = nn.L1Loss()
            elif loss_func == 'CE':
                self.loss_func = nn.CrossEntropyLoss()
            elif loss_func == 'BCE':
                self.loss_func = nn.BCEWithLogitsLoss()
            else:
                raise NotImplementedError(f'Unknown loss string: {loss_func}')
        else:
            self.loss_func = loss_func

    def forward(self, image):
        return self.transformer(image)

    def training_step(self, batch, batch_idx):
        image, gt = batch
        pred = self.transformer(image)
        loss = self.loss_func(pred, gt)

        self.log('Training/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_func(pred, y)

        self.log('Validation/loss', loss)
        self.log('val_loss', loss, prog_bar=True, logger=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser
