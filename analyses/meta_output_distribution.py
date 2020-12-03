import os
from argparse import ArgumentParser
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

    n_imgs = 4

    task = ImageTransformTask.load_from_checkpoint(checkpoint_path,
                                                   transformer=ShallowResNetMeta(in_channels=3, out_channels=3))
    data = ImageTransformDataModule(**vars(args))
    data.setup()
    task.freeze()
    parameterizing_cnn = task.transformer.parameterizing.to(device)
    parameterized_cnn = task.transformer.parameterized.to(device)

    dataloader = iter(data.val_dataloader(shuffle=True))
    x, y = next(dataloader)
    x, y = x.to(device)[:n_imgs], y.to(device)[:n_imgs]
    weights, biases = parameterizing_cnn(x)
    yhat = parameterized_cnn(x, weights, biases)

    images = []
    for i in range(len(x)):
        images.append(y[i])
        images.append(yhat[i])

        xi = x[i].unsqueeze(dim=0).repeat(len(x) - 1, 1, 1, 1)
        wrong_weight = [w[[j for j in range(len(x)) if j != i]] for w in weights]
        wrong_bias = [b[[j for j in range(len(x)) if j != i]] for b in biases]
        wrong_yhat = parameterized_cnn(xi, wrong_weight, wrong_bias)
        for j in range(len(wrong_yhat)):
            images.append(wrong_yhat[j])

    weight_images = make_grid(images, nrow=len(x) + 1, normalize=True, range=(-1, 1)).cpu()
    weight_images = to_pil_image(weight_images)
    weight_images = resize(weight_images, 700, interpolation=Image.NEAREST)
    weight_images.save('analyses/results/meta_output_samples.jpg')


parser = ArgumentParser()
parser.add_argument('--run_name', type=str, default=None)
parser = ImageTransformDataModule.add_data_specific_args(parser)
args = parser.parse_args()

main(args)
