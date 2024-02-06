import argparse
import json
import os
import sys
import torch
import tqdm
import numpy as np

from models.generator import Generator

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

import torchvision.utils as torch_utils
import torch

def get_args():
    parser = argparse.ArgumentParser(description="Inference script for GAN/SAN",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--params", type=str, default="./hparams/params.json", help="path to hyperparameters")
    parser.add_argument("--model", type=str, default="gan", help="model's name / 'gan' or 'san'")
    parser.add_argument('--disable_class', action='store_true', help='disable class conditioning')
    parser.add_argument("--logdir", type=str, default="./logs", help="directory storing log files")
    parser.add_argument("--num_samples_per_class", type=int, default=8,
                        help="number of samples to generate during test")
    parser.add_argument("--ffmpeg", type=str, required=True, help="ffmpeg path")
    parser.add_argument("--animation", type=str, required=True, help="path to save animation (png file)")

    return parser.parse_args()


def main(args):
    matplotlib.rcParams['animation.ffmpeg_path'] = args.ffmpeg

    with open(args.params, "r") as f:
        params = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model_name = args.model
    if not model_name in ['gan', 'san']:
        raise RuntimeError("A model name have to be 'gan' or 'san'.")

    # dataloading
    num_class = 10

    # model
    use_class = not args.disable_class
    generator = Generator(params["dim_latent"], num_class=num_class if use_class else 0)
    generator = generator.to(device)


    ckpt_dir = f'{args.logdir}/{model_name}/'
    if str(device) != 'cpu':
        generator.torch.load_state_dict(torch.load(ckpt_dir + model_name + ".generator.pt"))
    else:
        generator.load_state_dict(torch.load(ckpt_dir + model_name + ".generator.pt",
                                             map_location=torch.device('cpu')))

    animation_save_path = os.path.join(args.animation)
    if not os.path.exists(os.path.dirname(animation_save_path)) and len(os.path.dirname(animation_save_path)) > 0:
        os.makedirs(os.path.dirname(animation_save_path))

    print(generator)
    generator.eval()

    msg = ["\t{0}: {1}".format(key, val) for key, val in params.items()]
    print("hyperparameters: \n" + "\n".join(msg))

    progress = list()
    # inference / generate new samples
    with torch.no_grad():
        latent = torch.randn(args.num_samples_per_class * num_class, params["dim_latent"]).to(device)
        class_ids = torch.arange(num_class, dtype=torch.long,
                                    device=device).repeat_interleave(args.num_samples_per_class)
        imgs_fake = generator(latent, class_ids)
        progress.append(torch_utils.make_grid(imgs_fake, padding=2, normalize=True))

    # Progress Animation
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    for i in range(len(progress)):
        plt.imshow(np.transpose(progress[i], (1, 2, 0)), animated=True)
    plt.savefig(animation_save_path, format='png')
    plt.close() 

if __name__ == '__main__':
    args = get_args()
    main(args)
