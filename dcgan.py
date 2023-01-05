import argparse
import os
import numpy as np
import math
import snntorch as snn

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.etqdm import etqdm

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("-s", "--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class GenMid(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_size = opt.img_size // 4
        self.Num = 200
        self.l1 = nn.Linear(opt.latent_dim, 128 * self.init_size ** 2)
        self.bn1 = nn.BatchNorm2d(128)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(self.Num * 128, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128, 0.8)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64, 0.8)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, opt.channels, 3, stride=1, padding=1)
        self.fc_last = nn.Linear(1024, 1024)
        self.conv_last = nn.Conv2d(self.Num, 1, 3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        self.lif1 = snn.Leaky(beta=0.95)

    def forward(self, z):
        B = z.shape[0]
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out = self.bn1(out)
        out = self.up1(out)
        mem = self.lif1.init_leaky()
        spk_rec = []
        mem_rec = []
        for _ in range(self.Num):
            spk, mem = self.lif1(out, mem)
            spk_rec.append(spk)
            mem_rec.append(mem)
        out = torch.stack(spk_rec, dim=0).reshape(B, -1, 16, 16)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu1(out)
        out = self.up2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.lrelu2(out)
        out = self.conv3(out)



        # out = self.conv_last(spk_rec.transpose(0, 1).reshape(B, -1, 32, 32))
        # out = F.relu(out)
        # out = self.fc_last(out.view(B, -1))
        # out = out.view(B, 1, 32, 32)
        out = self.tanh(out)
        return out


class Gen2(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_size = opt.img_size // 4
        self.Num = 100
        self.l1 = nn.Linear(opt.latent_dim, 128 * self.init_size ** 2)
        self.bn1 = nn.BatchNorm2d(128)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128, 0.8)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(self.Num * 128, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64, 0.8)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, opt.channels, 3, stride=1, padding=1)
        self.fc_last = nn.Linear(1024, 1024)
        self.conv_last = nn.Conv2d(self.Num, 1, 3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        self.lif1 = snn.Leaky(beta=0.95)

    def forward(self, z):
        B = z.shape[0]
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out = self.bn1(out)
        out = self.up1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu1(out)
        out = self.up2(out)
        
        mem = self.lif1.init_leaky()
        spk_rec = []
        mem_rec = []
        for _ in range(self.Num):
            spk, mem = self.lif1(out, mem)
            spk_rec.append(spk)
            mem_rec.append(mem)
        out = torch.stack(spk_rec, dim=0).reshape(B, -1, 32, 32)
        
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.lrelu2(out)
        out = self.conv3(out)

        # out = self.conv_last(spk_rec.transpose(0, 1).reshape(B, -1, 32, 32))
        # out = F.relu(out)
        # out = self.fc_last(out.view(B, -1))
        # out = out.view(B, 1, 32, 32)
        out = self.tanh(out)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Linear(opt.latent_dim, 128 * self.init_size ** 2)
        self.bn1 = nn.BatchNorm2d(128)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128, 0.8)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64, 0.8)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, opt.channels, 3, stride=1, padding=1)
        self.Num = 200
        self.fc_last = nn.Linear(1024, 1024)
        self.conv_last = nn.Conv2d(self.Num, 1, 3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        self.lif1 = snn.Leaky(beta=0.95)

    def forward(self, z):
        B = z.shape[0]
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out = self.bn1(out)
        out = self.up1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu1(out)
        out = self.up2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.lrelu2(out)
        out = self.conv3(out)

        mem = self.lif1.init_leaky()
        spk_rec = []
        mem_rec = []
        for _ in range(self.Num):
            spk, mem = self.lif1(out, mem)
            spk_rec.append(spk)
            mem_rec.append(mem)
        spk_rec = torch.stack(spk_rec, dim=0) # [N, B, C, H, W]
        mem_rec = torch.stack(mem_rec, dim=0)

        # out = self.conv_last(spk_rec.transpose(0, 1).reshape(B, -1, 32, 32))
        # out = F.relu(out)
        # out = self.fc_last(out.view(B, -1))
        # out = out.view(B, 1, 32, 32)
        out = self.tanh(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, args=None):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(opt.channels, 16, 3, 2, 1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout2 = nn.Dropout2d(0.25)
        self.bn2 = nn.BatchNorm2d(32, 0.8)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout3 = nn.Dropout2d(0.25)
        self.bn3 = nn.BatchNorm2d(64, 0.8)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout4 = nn.Dropout2d(0.25)
        self.bn4 = nn.BatchNorm2d(128, 0.8)
        # snn lif
        self.lif1 = snn.Leaky(beta=0.95)


        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.Num = 50
        self.adv_layer = nn.Sequential(nn.Linear(self.Num * 128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        B = img.shape[0]
        out = self.conv1(img)
        out = self.lrelu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.lrelu2(out)
        out = self.dropout2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.lrelu3(out)
        out = self.dropout3(out)
        out = self.bn3(out)
        out = self.conv4(out)
        out = self.lrelu4(out)
        out = self.dropout4(out)
        out = self.bn4(out)
        
        mem = self.lif1.init_leaky()
        spk_rec = []
        mem_rec = []
        for _ in range(self.Num):
            spk, mem = self.lif1(out, mem)
            spk_rec.append(spk)     
            mem_rec.append(mem)
        spk_rec = torch.stack(spk_rec, dim=0) # [N, B, C, H, W]
        mem_rec = torch.stack(mem_rec, dim=0)

        out = spk_rec.transpose(0, 1).reshape(B, -1)
        validity = self.adv_layer(out)
        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Gen2()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------
        for _ in range(1):
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/dcgan/gan.png", nrow=5, normalize=True)
