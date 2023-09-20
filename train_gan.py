import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm

from models.discriminator import Discriminator
from models.generator import Generator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def update_discriminator(x, class_ids, discriminator, generator, optimizer, params):
    bs = x.size(0)
    device = x.device

    optimizer.zero_grad()

    # predictions on generating distribution
    loss_real = discriminator(x, class_ids, loss_type='D_real')

    # predictions on fake distribution
    latent = torch.randn(bs, params["dim_latent"], device=device)
    batch_fake = generator(latent, class_ids)
    loss_fake = discriminator(batch_fake.detach(), class_ids, loss_type='D_fake')

    loss_d = (loss_real + loss_fake) / 2.
    loss_d.backward()
    optimizer.step()

    return loss_real.detach(), loss_fake.detach()


def update_generator(num_class, discriminator, generator, optimizer, params, device):
    optimizer.zero_grad()

    bs = params['batch_size']
    latent = torch.randn(bs, params["dim_latent"], device=device)

    class_ids = torch.randint(num_class, size=(bs,), device=device)
    batch_fake = generator(latent, class_ids)

    loss_g = discriminator(batch_fake, class_ids, loss_type='G')
    loss_g.backward()
    optimizer.step()

    return loss_g.detach()


def test_discriminator(test_loader, discriminator, device):
    loss = 0.
    for x, class_ids in tqdm.tqdm(test_loader):
        x = x.to(device)

        # predictions on generating distribution
        loss_ = discriminator(x, class_ids, loss_type='D_real')
        loss += loss_

    return loss / len(test_loader)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, required=True, help="path to MNIST dataset folder")
    parser.add_argument("--params", type=str, default="./hparams/params.json", help="path to hyperparameters")
    parser.add_argument("--model", type=str, default="gan", help="model's name / 'gan' or 'san'")
    parser.add_argument('--disable_class', action='store_true', help='disable class conditioning')
    parser.add_argument("--logdir", type=str, default="./logs", help="directory storing log files")
    parser.add_argument("--device", type=int, default=0, help="gpu device to use")
    parser.add_argument("--num_samples_per_class", type=int, default=8,
                        help="number of samples to generate during test")
    parser.add_argument("--jobname", type=str, default="none", help="job/directory name used for tensorboard outputs")

    return parser.parse_args()


def main(args):
    with open(args.params, "r") as f:
        params = json.load(f)

    device = f'cuda:{args.device}' if args.device is not None else 'cpu'
    model_name = args.model
    if not model_name in ['gan', 'san']:
        raise RuntimeError("A model name have to be 'gan' or 'san'.")

    # dataloading
    num_class = 10
    train_dataset = datasets.MNIST(root=args.datadir, transform=transforms.ToTensor(), train=True)
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], num_workers=4,
                              pin_memory=True, persistent_workers=True, shuffle=True)
    test_dataset = datasets.MNIST(root=args.datadir, transform=transforms.ToTensor(), train=False)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], num_workers=4,
                             pin_memory=True, persistent_workers=True, shuffle=False)

    # model
    use_class = not args.disable_class
    generator = Generator(params["dim_latent"], num_class=num_class if use_class else 0)
    discriminator = Discriminator(model_type=model_name, num_class=num_class if use_class else 0)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # optimizer
    betas = (params["beta_1"], params["beta_2"])
    optimizer_G = optim.Adam(generator.parameters(), lr=params["learning_rate"], betas=betas)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=params["learning_rate"], betas=betas)

    ckpt_dir = f'{args.logdir}/{model_name}/'
    tb_dir = f'{args.logdir}/tensorboard/{model_name}_{args.jobname}'
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    if not os.path.exists(tb_dir):
        os.mkdir(tb_dir)
    writer = SummaryWriter(tb_dir)

    steps_per_epoch = len(train_loader)

    msg = ["\t{0}: {1}".format(key, val) for key, val in params.items()]
    print("hyperparameters: \n" + "\n".join(msg))

    # eval initial states
    with torch.no_grad():
        latent = torch.randn(args.num_samples_per_class * num_class, params["dim_latent"]).cuda()
        class_ids = torch.arange(num_class, dtype=torch.long,
                                 device=device).repeat_interleave(args.num_samples_per_class)
        imgs_fake = generator(latent, class_ids)

        # test discriminator
        loss_test = test_discriminator(test_loader, discriminator, device)

        writer.add_scalar("loss_test/D_real", loss_test, 0)
        writer.add_images("generated fake images", imgs_fake, 0)

    # main training loop
    for n in range(params["num_epochs"]):
        loader = iter(train_loader)

        print("epoch: {0}/{1}".format(n + 1, params["num_epochs"]))
        for i in tqdm.trange(steps_per_epoch):
            x, class_ids = next(loader)
            x = x.to(device)
            class_ids = class_ids.to(device)

            loss_real, loss_fake = update_discriminator(x, class_ids, discriminator, generator, optimizer_D, params)
            loss_G = update_generator(num_class, discriminator, generator, optimizer_G, params, device)

            writer.add_scalar("loss_train/D_real", loss_real, i + n * steps_per_epoch)
            writer.add_scalar("loss_train/D_fake", loss_fake, i + n * steps_per_epoch)
            writer.add_scalar("loss_train/G", loss_G, i + n * steps_per_epoch)

        torch.save(generator.state_dict(), ckpt_dir + model_name + ".g." + str(n) + ".tmp")
        torch.save(discriminator.state_dict(), ckpt_dir + model_name + ".d." + str(n) + ".tmp")

        # eval
        with torch.no_grad():
            latent = torch.randn(args.num_samples_per_class * num_class, params["dim_latent"]).cuda()
            class_ids = torch.arange(num_class, dtype=torch.long,
                                     device=device).repeat_interleave(args.num_samples_per_class)
            imgs_fake = generator(latent, class_ids)

            # test discriminator
            loss_test = test_discriminator(test_loader, discriminator, device)

            writer.add_scalar("loss_test/D_real", loss_test, n + 1)
            writer.add_images("generated fake images", imgs_fake, n + 1)
            # del latent, imgs_fake

    writer.close()

    torch.save(generator.state_dict(), ckpt_dir + model_name + ".generator.pt")
    torch.save(discriminator.state_dict(), ckpt_dir + model_name + ".discriminator.pt")


if __name__ == '__main__':
    args = get_args()
    main(args)
