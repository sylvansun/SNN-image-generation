import snntorch as snn
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,
                 noise_dim=100,
                 gen_num_hidden1=128,
                 gen_num_hidden2=256,
                 gen_num_hidden3=512,
                 gen_num_hidden4=1024,
                 gen_num_outputs=28*28,
                 num_steps=25,
                 beta=0.95):
        super().__init__()
        self.num_steps = num_steps
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

        for step in range(self.num_steps):
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
    def __init__(self,
                 dis_inputs=28*28,
                 dis_num_hidden1=512,
                 dis_num_hidden2=256,
                 dis_num_outputs=1,
                 num_steps=25,
                 beta=0.95):
        super().__init__()
        self.num_steps = num_steps
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

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
        
        return torch.stack(mem3_rec, dim=0)