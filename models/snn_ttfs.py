import snntorch as snn
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,
                 noise_dim = 100,
                 gen_num_hidden1 = 100,
                 gen_num_hidden2 = 400,
                 gen_num_outputs = 28*28,
                 num_steps = 25,
                 beta = 0.95,
                 reward_scale = 2):
        super().__init__()
        self.num_steps  = num_steps
        self.reward_scale = reward_scale
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

        for step in range(self.num_steps):
            cur1 = self.reward_scale * self.fc1(x[step])
            cur1 = self.relu(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.reward_scale * self.fc2(spk1)
            cur2 = self.relu(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.reward_scale * self.fc3(spk2)
            cur3 = self.relu(cur3)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(self.sigmoid(mem3))

        return torch.stack(spk3_rec, dim=1) # bz, 25, 28*28

class Discriminator(nn.Module):
    def __init__(self,
                 dis_inputs = 28*28,
                 dis_num_hidden1 = 400,
                 dis_num_hidden2 = 100,
                 dis_num_outputs = 1,
                 num_steps = 25,
                 beta = 0.95,
                 reward_scale = 2):
        super().__init__()
        self.num_steps = num_steps
        self.reward_scale = reward_scale
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

        for step in range(self.num_steps):
            cur1 = self.reward_scale * self.fc1(x[step])
            cur1 = self.relu(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.reward_scale * self.fc2(spk1)
            cur2 = self.relu(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.reward_scale * self.fc3(spk2)
            cur3 = self.relu(cur3)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
        
        return torch.stack(mem3_rec, dim=1)