import snntorch as snn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

n_epochs = 200
batch_size = 64
sample_interval = 400
clip_value = 0.01
n_critic = 1
data_path='./mnist'
device = torch.device('cuda:1')
dtype = torch.float
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

noise_dim = 100
gen_num_hidden1 = 128
gen_num_hidden2 = 256
gen_num_hidden3 = 512
gen_num_hidden4 = 1024
gen_num_outputs = 28*28

dis_inputs = 28*28
dis_num_hidden1 = 512
dis_num_hidden2 = 256
dis_num_outputs = 1

num_steps = 25
start_grad_step = 24
beta = 0.95

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(noise_dim, gen_num_hidden1)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(gen_num_hidden1, gen_num_hidden2)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(gen_num_hidden2, gen_num_hidden3)
        self.lif3 = snn.Leaky(beta=beta)
        self.fc4 = nn.Linear(gen_num_hidden3, gen_num_hidden4)
        self.lif4 = snn.Leaky(beta=beta)
        self.fc5 = nn.Linear(gen_num_hidden4, gen_num_outputs)
        self.lif5 = snn.Leaky(beta=beta)
        self.tanh = nn.Tanh()

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        spk5_rec = []
        mem5_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            cur5 = self.fc5(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)
            spk5_rec.append(spk5)
            mem5_rec.append(self.tanh(mem5))

        return torch.stack(mem5_rec, dim=0)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(dis_inputs, dis_num_hidden1)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(dis_num_hidden1, dis_num_hidden2)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(dis_num_hidden2, dis_num_outputs)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []
        mem3_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
        
        return torch.stack(mem3_rec, dim=0)

generator = Generator().to(device=device, dtype=dtype)
discriminator = Discriminator().to(device=device, dtype=dtype)
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=5e-5)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=5e-5)

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        real_imgs = imgs.to(device=device, dtype=dtype)
        z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], noise_dim))).to(device=device, dtype=dtype)

        optimizer_D.zero_grad()
        gen_imgs = generator(z)
        gen_imgs = gen_imgs.detach()
        d_loss = 0
        for step in range(start_grad_step, num_steps):
            real_loss = torch.mean(-discriminator(real_imgs.reshape(batch_size, -1)))
            fake_loss = torch.mean(discriminator(gen_imgs[step]))
            d_loss += (fake_loss + real_loss)
        d_loss = d_loss / (num_steps - start_grad_step)
        d_loss.backward()
        optimizer_D.step()

        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)

        if i % n_critic == 0:
            optimizer_G.zero_grad()
            gen_imgs = generator(z)
            g_loss = 0
            for step in range(start_grad_step, num_steps):
                g_loss -= torch.mean(discriminator(gen_imgs[step]))
            g_loss = g_loss / (num_steps - start_grad_step)
            g_loss.backward()
            optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(train_loader), d_loss.item() / 2, g_loss.item())
        )

        batches_done = epoch * len(train_loader) + i
        if batches_done % sample_interval == 0:
            image = gen_imgs.data[:, 0, ].reshape([num_steps, 1, 28, 28])
            save_image(image, "images/%d.png" % batches_done, nrow=5, normalize=True,)

