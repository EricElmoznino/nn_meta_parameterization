import os
import shutil
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tasks import ImageTransformTask
from datasets import Shoe2EdgeDataModule
from tasks.custom_callbacks import TensorBoardImageCallback
from models.baselines import ShallowCNN
from models.meta_cnns import OneLayerResNetMeta


def dispatch_model(model_name):
    if model_name == 'baseline':
        return ShallowCNN(in_channels=3, out_channels=1)
    elif model_name == 'meta':
        return OneLayerResNetMeta(in_channels=3, out_channels=1)
    else:
        raise ValueError(f'Unknown model: {model_name}')


parser = ArgumentParser()
parser.add_argument('--run_name', type=str, default='debug')
parser.add_argument('--data_dir', type=str, default='/home/eric/datasets/shoe2edge')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model', type=str, default='meta')
parser = ImageTransformTask.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

save_dir = os.path.join('saved_runs', args.run_name)
if args.run_name == 'debug':
    shutil.rmtree(save_dir, ignore_errors=True)
    overfit_batches = 1
else:
    overfit_batches = 0
os.mkdir(save_dir)

model = dispatch_model(args.model)
data = Shoe2EdgeDataModule(data_dir=args.data_dir)
task = ImageTransformTask(model, **vars(args))

trainer = Trainer.from_argparse_args(args, default_root_dir=save_dir, overfit_batches=overfit_batches,
                                     callbacks=[ModelCheckpoint(monitor='Validation/loss'),
                                                EarlyStopping(monitor='Validation/loss'),
                                                TensorBoardImageCallback(n_cols=2)])
trainer.fit(task, datamodule=data)
