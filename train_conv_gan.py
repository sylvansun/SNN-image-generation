import os
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np

from models.snn_conv_gan import Discriminator
from models.ann_gan import Generator
from utils.dataset import get_dataset
from snntorch import functional as SF
from utils.etqdm import etqdm


def main(args):
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    sample_interval = args.sample_interval if not args.vis else 1
    device = torch.device(f"cuda:{args.gpu}")
    num_steps = args.num_steps
    start_grad_step = args.start_grad_step
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    vis = args.vis
    dtype = torch.float
    noise_dim = 100
    num_vis = 25

    train_loader, _ = get_dataset(batch_size, "mnist")

    generator = Generator(noise_dim=noise_dim).to(device=device, dtype=dtype)
    discriminator = Discriminator(num_steps=num_steps).to(device=device, dtype=dtype)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=5e-3)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=3e-2)
    loss = SF.ce_rate_loss()

    for epoch in range(n_epochs):
        train_bar = etqdm(train_loader, desc=f"Epoch {epoch:03d}")
        for i, (imgs, _) in enumerate(train_bar):
            valid = torch.ones((imgs.shape[0])).to(device=device, dtype=torch.int64)
            fake = torch.zeros((imgs.shape[0])).to(device=device, dtype=torch.int64)
            real_imgs = imgs.to(device=device, dtype=dtype)
            z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], noise_dim))).to(device=device, dtype=dtype) #BS*noise_dim
            for _ in range(1):
                optimizer_G.zero_grad()
                gen_imgs, _ = generator(z) # num_steps * BS * 28^2
                # print(spk.max(), spk.min())
                # print(gen_imgs.max(), gen_imgs.min())
                g_loss = 0
                for step in range(start_grad_step, num_steps):
                    pred, _ = discriminator(gen_imgs)
                    # pred = torch.cat([1 - pred, pred], dim=2)
                    g_loss += loss(pred, valid)
                g_loss = g_loss / (num_steps - start_grad_step)
                g_loss.backward()
                optimizer_G.step()

            optimizer_D.zero_grad()
            gen_imgs = gen_imgs.detach()
            d_loss = 0
            for step in range(start_grad_step, num_steps):
                real_pred, _ = discriminator(real_imgs.reshape(batch_size, -1))
                fake_pred, _ = discriminator(gen_imgs)
                # real_pred = torch.cat([1 - real_pred, real_pred], dim=2)
                # fake_pred = torch.cat([1 - fake_pred, fake_pred], dim=2)
                real_loss = loss(real_pred, valid)
                fake_loss = loss(fake_pred, fake)
                d_loss += (fake_loss + real_loss) / 2
            d_loss = d_loss / (num_steps - start_grad_step)
            d_loss.backward()
            optimizer_D.step()

            train_bar.set_postfix_str(f"D_loss: {d_loss.item() / 2:.4f}, G_loss: {g_loss.item():.4f}")
            batches_done = epoch * len(train_loader) + i
            if batches_done % sample_interval == 0:
                image = gen_imgs.data[ :num_vis].reshape([num_vis, 1, 28, 28])
                real_image = real_imgs.data[:25, 0].reshape([-1, 1, 28, 28])
                image = torch.cat([real_image, image], dim=0)
                if vis:
                    save_image(image, os.path.join(output_dir, "convgan.png"), nrow=5, normalize=True)
                else:
                    save_image(image, os.path.join(output_dir, "convgan_%d.png" % batches_done), nrow=5, normalize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-e", "--n_epochs", type=int, default=200)
    parser.add_argument("-n", "--num_steps", type=int, default=25)
    parser.add_argument("-s", "--start_grad_step", type=int, default=24)
    parser.add_argument("-i", "--sample_interval", type=int, default=400)
    parser.add_argument("-o", "--output_dir", type=str, default="images/convgan")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    main(args)
