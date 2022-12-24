import torch
from torch.utils.data import DataLoader
import numpy as np

from utils.dataset import TTFSDataset, TTFS
from models.snn_ttfs import Generator, Discriminator

def main():
    n_epochs = 200
    batch_size = 64
    sample_interval = 400
    clip_value = 0.1
    device = torch.device('cuda:1')
    dtype = torch.float
    mnist_train = TTFSDataset()
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

    noise_dim = 100
    num_steps = 25

    generator = Generator(noise_dim=noise_dim,num_steps=num_steps).to(device=device, dtype=dtype)
    discriminator = Discriminator(num_steps=num_steps).to(device=device, dtype=dtype)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    ttfs = TTFS(step_num=num_steps, device=device)

    for epoch in range(n_epochs):
        for i, imgs in enumerate(train_loader):
            weight = 2 * (torch.arange(0, num_steps).to(dtype=torch.float) / (num_steps - 1) - 0.5)
            weight = weight.to(device)
            real_imgs = imgs.to(device=device, dtype=dtype)

            z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], num_steps, noise_dim))).to(device=device, dtype=dtype)

            optimizer_G.zero_grad()
            gen_imgs = generator(z)
            g_loss = -torch.sum(weight @ discriminator(gen_imgs)) / batch_size
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            gen_imgs = gen_imgs.detach()
            real_loss = -torch.sum(weight @ discriminator(real_imgs)) / batch_size
            fake_loss = torch.sum(weight @ discriminator(gen_imgs)) / batch_size
            d_loss = (fake_loss + real_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)
            for q in generator.parameters():
                q.data.clamp_(-clip_value, clip_value)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(train_loader) + i
            if batches_done % sample_interval == 0:
                ttfs_image = gen_imgs.data[0]
                ttfs.TTFS_to_image(ttfs=ttfs_image, batches_done=batches_done)

if __name__ == "__main__":
    main()