import torch
import torch.nn as nn
import snntorch as snn


class GenFront(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.init_size = args.img_size // 4
        self.Num = 200
        self.l1 = nn.Linear(args.latent_dim, 128 * self.init_size ** 2)
        self.bn1 = nn.BatchNorm2d(128)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(self.Num * 128, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128, 0.8)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64, 0.8)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, args.channels, 3, stride=1, padding=1)
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


class GenMid(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.init_size = args.img_size // 4
        self.Num = 100
        self.l1 = nn.Linear(args.latent_dim, 128 * self.init_size ** 2)
        self.bn1 = nn.BatchNorm2d(128)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128, 0.8)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(self.Num * 128, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64, 0.8)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, args.channels, 3, stride=1, padding=1)
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


class GenBack(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.init_size = args.img_size // 4
        self.l1 = nn.Linear(args.latent_dim, 128 * self.init_size ** 2)
        self.bn1 = nn.BatchNorm2d(128)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128, 0.8)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64, 0.8)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, args.channels, 3, stride=1, padding=1)
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

class Gen(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.init_size = args.img_size // 4
        self.l1 = nn.Linear(args.latent_dim, 128 * self.init_size ** 2)
        self.bn1 = nn.BatchNorm2d(128)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128, 0.8)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64, 0.8)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, args.channels, 3, stride=1, padding=1)
        self.tanh = nn.Tanh()

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
        out = self.tanh(out)
        return out


class DisSpike(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.conv1 = nn.Conv2d(args.channels, 16, 3, 2, 1)
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
        ds_size = args.img_size // 2 ** 4
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
    
    
class Dis(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.conv1 = nn.Conv2d(args.channels, 16, 3, 2, 1)
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


        # The height and width of downsampled image
        ds_size = args.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

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
        out = out.view(B, -1)
        validity = self.adv_layer(out)
        return validity