import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import itertools
from helpers import TTFSDataset, TTFS

n_epochs = 200
batch_size = 64
sample_interval = 400
clip_value = 0.1
n_critic = 1
reward_scale = 2
data_path='./mnist'
device = torch.device('cuda:1')
dtype = torch.float
mnist_train = TTFSDataset()
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

noise_dim = 100
gen_num_hidden1 = 100
gen_num_hidden2 = 400
gen_num_outputs = 28*28

dis_inputs = 28*28
dis_num_hidden1 = 400
dis_num_hidden2 = 100
dis_num_outputs = 1

num_steps = 25
beta = 0.95

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(noise_dim, gen_num_hidden1)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(gen_num_hidden1, gen_num_hidden2)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(gen_num_hidden2, gen_num_outputs)
        self.lif3 = snn.Leaky(beta=beta)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.permute(x, (1, 0, 2))
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []
        mem3_rec = []

        for step in range(num_steps):
            cur1 = reward_scale * self.fc1(x[step])
            cur1 = self.relu(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = reward_scale * self.fc2(spk1)
            cur2 = self.relu(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = reward_scale * self.fc3(spk2)
            cur3 = self.relu(cur3)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(self.sigmoid(mem3))

        return torch.stack(spk3_rec, dim=1) # bz, 25, 28*28

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(dis_inputs, dis_num_hidden1)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(dis_num_hidden1, dis_num_hidden2)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(dis_num_hidden2, dis_num_outputs)
        self.lif3 = snn.Leaky(beta=beta)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.permute(x, (1, 0, 2))
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []
        mem3_rec = []

        for step in range(num_steps):
            cur1 = reward_scale * self.fc1(x[step])
            cur1 = self.relu(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = reward_scale * self.fc2(spk1)
            cur2 = self.relu(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = reward_scale * self.fc3(spk2)
            cur3 = self.relu(cur3)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
        
        return torch.stack(mem3_rec, dim=1)

generator = Generator().to(device=device, dtype=dtype)
discriminator = Discriminator().to(device=device, dtype=dtype)
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

