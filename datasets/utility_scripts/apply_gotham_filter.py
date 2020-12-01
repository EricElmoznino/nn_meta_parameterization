# Taken from https://github.com/lukexyz/CV-Instagram-Filters/
from argparse import ArgumentParser
import os
import skimage
from skimage import io, filters
import numpy as np
from tqdm import tqdm


def main(args):
    files = os.listdir(args.data_dir)
    for file in tqdm(files):
        original = skimage.io.imread(os.path.join(args.data_dir, file))
        original = skimage.util.img_as_float(original)

        # 1. Colour channel adjustment example
        r, g, b = split_image_into_channels(original)
        r_interp = channel_adjust(r, [0, 0.8, 1.0])
        red_channel_adj = merge_channels(r_interp, g, b)

        # 2. Mid tone colour boost
        r, g, b = split_image_into_channels(original)
        r_boost_lower = channel_adjust(r, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
        r_boost_img = merge_channels(r_boost_lower, g, b)

        # 3. Making the blacks bluer
        bluer_blacks = merge_channels(r_boost_lower, g, np.clip(b + 0.03, 0, 1.0))

        # 4. Sharpening the image
        sharper = sharpen(bluer_blacks, 1.3, 0.3)

        # 5. Blue channel boost in lower-mids, decrease in upper-mids
        r, g, b = split_image_into_channels(sharper)
        b_adjusted = channel_adjust(b, [0, 0.047, 0.118, 0.251, 0.318, 0.392, 0.42, 0.439, 0.475, 0.561, 0.58, 0.627, 0.671, 0.733, 0.847, 0.925, 1])
        gotham = merge_channels(r, g, b_adjusted)

        gotham = skimage.util.img_as_ubyte(gotham)
        file = file.replace('.', '_target.')
        skimage.io.imsave(os.path.join(args.data_dir, file), gotham)


def split_image_into_channels(image):
    """Look at each image separately"""
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    return red_channel, green_channel, blue_channel


def merge_channels(red, green, blue):
    """Merge channels back into an image"""
    return np.stack([red, green, blue], axis=2)


def sharpen(image, a, b):
    """Sharpening an image: Blur and then subtract from original"""
    blurred = skimage.filters.gaussian(image, sigma=10, multichannel=True)
    sharper = np.clip(image * a - blurred * b, 0, 1.0)
    return sharper


def channel_adjust(channel, values):
    # preserve the original size, so we can reconstruct at the end
    orig_size = channel.shape
    # flatten the image into a single array
    flat_channel = channel.flatten()

    # this magical numpy function takes the values in flat_channel
    # and maps it from its range in [0, 1] to its new squeezed and
    # stretched range
    adjusted = np.interp(flat_channel, np.linspace(0, 1, len(values)), values)

    # put back into the original image shape
    return adjusted.reshape(orig_size)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    main(args)
