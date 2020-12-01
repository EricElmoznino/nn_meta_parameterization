import os
import math
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import seaborn as sns; sns.set(style='darkgrid')
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, resize
from PIL import Image
from tasks import ImageTransformTask
from datasets import ImageTransformDataModule
from models.meta_cnns import OneLayerResNetMeta

torch.set_grad_enabled(False)
torch.manual_seed(27)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):
    run_name = args.run_name if args.run_name is not None else 'meta'
    checkpoint_dir = f'saved_runs/{run_name}/lightning_logs/version_0/checkpoints/'
    checkpoint_file = os.listdir(checkpoint_dir)[0]
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

    task = ImageTransformTask.load_from_checkpoint(checkpoint_path,
                                                   transformer=OneLayerResNetMeta(in_channels=3, out_channels=3))
    data = ImageTransformDataModule(**vars(args))
    data.setup()
    task.freeze()
    parameterizing_cnn = task.transformer.parameterizing.to(device)

    weights, biases = [], []
    for x, _ in tqdm(data.val_dataloader(shuffle=True)):
        x = x.to(device)
        weight, bias = parameterizing_cnn(x)
        weights.append(weight)
        biases.append(bias)
    weights, biases = torch.cat(weights), torch.cat(biases)

    intersample_weights_std = weights.std(dim=0).view(-1).cpu().numpy()
    intersample_biases_std = biases.std(dim=0).view(-1).cpu().numpy()
    intersample_weights_std = pd.DataFrame({'σ': intersample_weights_std, 'parameter': 'weights', 'distribution': 'inter-sample'})
    intersample_biases_std = pd.DataFrame({'σ': intersample_biases_std, 'parameter': 'biases', 'distribution': 'inter-sample'})

    intrasample_weights_std = weights.std(dim=[1, 2, 3, 4]).view(-1).cpu().numpy()
    intrasample_biases_std = biases.std(dim=1).view(-1).cpu().numpy()
    intrasample_weights_std = pd.DataFrame({'σ': intrasample_weights_std, 'parameter': 'weights', 'distribution': 'intra-sample'})
    intrasample_biases_std = pd.DataFrame({'σ': intrasample_biases_std, 'parameter': 'biases', 'distribution': 'intra-sample'})

    data = pd.concat([intersample_weights_std, intersample_biases_std, intrasample_weights_std, intrasample_biases_std])
    sns.violinplot(x='parameter', y='σ', hue='distribution', split=True, data=data)
    plt.tight_layout()
    plt.savefig('analyses/results/meta_param_distribution.jpg')

    n_imgs = 36
    weight_images = weights[:n_imgs, 0, 0:1]
    weight_images = make_grid(weight_images, nrow=int(math.sqrt(n_imgs)), normalize=True).cpu()[0]
    weight_images = to_pil_image(weight_images)
    weight_images = resize(weight_images, 700, interpolation=Image.NEAREST)
    weight_images.save('analyses/results/meta_param_samples.jpg')


parser = ArgumentParser()
parser.add_argument('--run_name', type=str, default=None)
parser = ImageTransformDataModule.add_data_specific_args(parser)
args = parser.parse_args()

main(args)
