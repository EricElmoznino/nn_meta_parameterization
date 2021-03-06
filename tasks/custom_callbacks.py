import torch
from torch.nn import functional as F
from pytorch_lightning import Callback
from torchvision.utils import make_grid
import math


class TensorBoardImageCallback(Callback):

    def __init__(self, n_imgs=16, n_cols=None, val_only=False, every_n_epochs=5):
        self.n_imgs = n_imgs
        self.n_cols = int(math.sqrt(n_imgs)) if n_cols is None else n_cols
        self.val_only = val_only
        self.every_n_epochs = every_n_epochs

        self.train_saved = True
        self.val_saved = True
        self.test_saved = True

    def on_train_epoch_start(self, trainer, pl_module):
        if (trainer.current_epoch - 1) % self.every_n_epochs == 0:
            self.train_saved = False

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0 and not trainer.running_sanity_check:
            self.val_saved = False

    def on_test_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.test_saved = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if not self.train_saved:
            self.log_images(pl_module, batch, label='Training', epoch=trainer.current_epoch - 1)
            self.train_saved = True

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if not self.val_saved:
            self.log_images(pl_module, batch, label='Validation', epoch=trainer.current_epoch)
            self.val_saved = True

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if not self.test_saved:
            self.log_images(pl_module, batch, label='Test', epoch=trainer.current_epoch)
            self.test_saved = True

    def log_images(self, pl_module, batch, label, epoch):
        label = '' if label is None else label + '/'

        x, y = self.decompose_batch(batch)
        x, y, yhat = self.image_samples(pl_module, x, y)

        pl_module.logger.experiment.add_image(label + 'input', x, epoch)
        pl_module.logger.experiment.add_image(label + 'prediction', yhat, epoch)
        if y is not None:
            pl_module.logger.experiment.add_image(label + 'ground_truth', y, epoch)

    def image_samples(self, pl_module, x, y):
        x = x[:self.n_imgs].to(pl_module.device)
        with torch.no_grad():
            yhat = pl_module(x)

        x, yhat = x.cpu(), yhat.cpu()
        x, yhat = F.interpolate(x, size=64, mode='nearest'), F.interpolate(yhat, size=64, mode='nearest')
        x, yhat = make_grid(x, nrow=self.n_cols, normalize=True), \
                  make_grid(yhat, nrow=self.n_cols, normalize=True)
        if y is not None:
            y = y[:self.n_imgs].cpu()
            y = F.interpolate(y, size=64, mode='nearest')
            y = make_grid(y, nrow=self.n_cols, normalize=True)

        return x, y, yhat

    @staticmethod
    def decompose_batch(batch):
        assert isinstance(batch, torch.Tensor) or len(batch) == 2
        if isinstance(batch, torch.Tensor):
            return batch, None
        else:
            return batch
