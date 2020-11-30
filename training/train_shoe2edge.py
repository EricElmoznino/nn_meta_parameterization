import os
import shutil
from argparse import ArgumentParser
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tasks import ImageTransformTask
from datasets import Shoe2EdgeDataModule
from tasks.custom_callbacks import TensorBoardImageCallback
from models.baselines import ShallowCNN
from models.meta_cnns import OneLayerResNetMeta


def main(args):
    run_name = args.run_name if args.run_name is not None else args.model
    save_dir = os.path.join('saved_runs', run_name)
    if args.debug:
        save_dir += '_DEBUG'
        shutil.rmtree(save_dir, ignore_errors=True)
        overfit_batches = 1
    else:
        overfit_batches = 0
    os.mkdir(save_dir)

    gpus = 1 if torch.cuda.is_available() else 0

    model = dispatch_model(args.model)
    data = Shoe2EdgeDataModule(data_dir=args.data_dir)
    task = ImageTransformTask(model, **vars(args))

    trainer = Trainer.from_argparse_args(args, default_root_dir=save_dir, gpus=gpus,  overfit_batches=overfit_batches,
                                         callbacks=[ModelCheckpoint(monitor='Validation/loss'),
                                                    EarlyStopping(monitor='Validation/loss', patience=10),
                                                    TensorBoardImageCallback(n_cols=2)])
    trainer.fit(task, datamodule=data)


def dispatch_model(model_name):
    if model_name == 'baseline':
        return ShallowCNN(in_channels=3, out_channels=1)
    elif model_name == 'meta':
        return OneLayerResNetMeta(in_channels=3, out_channels=1)
    else:
        raise ValueError(f'Unknown model: {model_name}')


parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--model', type=str, default='meta')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--batch_size', type=int, default=32)
parser = ImageTransformTask.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

main(args)
