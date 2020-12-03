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
from models.meta_cnns import ShallowResNetMeta

torch.set_grad_enabled(False)
torch.manual_seed(27)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):
    run_name = args.run_name if args.run_name is not None else 'meta'
    checkpoint_dir = f'saved_runs/{run_name}/lightning_logs/version_0/checkpoints/'
    checkpoint_file = os.listdir(checkpoint_dir)[0]
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

    task = ImageTransformTask.load_from_checkpoint(checkpoint_path,
                                                   transformer=ShallowResNetMeta(in_channels=3, out_channels=3))
    data = ImageTransformDataModule(**vars(args))
    data.setup()
    task.freeze()
    parameterizing_cnn = task.transformer.parameterizing.to(device)

    wl1, wl2, bl1, bl2 = [], [], [], []
    for x, _ in tqdm(data.val_dataloader(shuffle=True)):
        x = x.to(device)
        weights, biases = parameterizing_cnn(x)
        wl1.append(weights[0])
        wl2.append(weights[1])
        bl1.append(biases[0])
        bl2.append(biases[1])
    wl1, wl2, bl1, bl2 = torch.cat(wl1), torch.cat(wl2), torch.cat(bl1), torch.cat(bl2)

    intersample_wl1_std = wl1.std(dim=0).view(-1).cpu().numpy()
    intersample_wl2_std = wl2.std(dim=0).view(-1).cpu().numpy()
    intersample_bl1_std = bl1.std(dim=0).view(-1).cpu().numpy()
    intersample_bl2_std = bl2.std(dim=0).view(-1).cpu().numpy()
    intersample_wl1_std = pd.DataFrame({'σ': intersample_wl1_std, 'parameter': 'weights (layer 1)', 'distribution': 'inter-sample'})
    intersample_wl2_std = pd.DataFrame({'σ': intersample_wl2_std, 'parameter': 'weights (layer 2)', 'distribution': 'inter-sample'})
    intersample_bl1_std = pd.DataFrame({'σ': intersample_bl1_std, 'parameter': 'biases (layer 1)', 'distribution': 'inter-sample'})
    intersample_bl2_std = pd.DataFrame({'σ': intersample_bl2_std, 'parameter': 'biases (layer 2)', 'distribution': 'inter-sample'})

    intrasample_wl1_std = wl1.std(dim=[1, 2, 3, 4]).view(-1).cpu().numpy()
    intrasample_wl2_std = wl2.std(dim=[1, 2, 3, 4]).view(-1).cpu().numpy()
    intrasample_bl1_std = bl1.std(dim=1).view(-1).cpu().numpy()
    intrasample_bl2_std = bl2.std(dim=1).view(-1).cpu().numpy()
    intrasample_wl1_std = pd.DataFrame({'σ': intrasample_wl1_std, 'parameter': 'weights (layer 1)', 'distribution': 'intra-sample'})
    intrasample_wl2_std = pd.DataFrame({'σ': intrasample_wl2_std, 'parameter': 'weights (layer 2)', 'distribution': 'intra-sample'})
    intrasample_bl1_std = pd.DataFrame({'σ': intrasample_bl1_std, 'parameter': 'biases (layer 1)', 'distribution': 'intra-sample'})
    intrasample_bl2_std = pd.DataFrame({'σ': intrasample_bl2_std, 'parameter': 'biases (layer 2)', 'distribution': 'intra-sample'})

    data = pd.concat([intersample_wl1_std, intersample_wl2_std, intersample_bl1_std, intersample_bl2_std,
                      intrasample_wl1_std, intrasample_wl2_std, intrasample_bl1_std, intrasample_bl2_std])
    sns.violinplot(x='parameter', y='σ', hue='distribution', split=True, data=data)
    plt.tight_layout()
    plt.savefig('analyses/results/meta_param_distribution.jpg')

    n_imgs = 36
    weight_images = wl1[:n_imgs, 0, 0:1]
    weight_images = make_grid(weight_images, nrow=int(math.sqrt(n_imgs)), normalize=True).cpu()[0]
    weight_images = to_pil_image(weight_images)
    weight_images = resize(weight_images, 700, interpolation=Image.NEAREST)
    weight_images.save('analyses/results/meta_param_samples.jpg')


parser = ArgumentParser()
parser.add_argument('--run_name', type=str, default=None)
parser = ImageTransformDataModule.add_data_specific_args(parser)
args = parser.parse_args()

main(args)
