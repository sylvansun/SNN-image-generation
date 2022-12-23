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
import pickle

class TTFSDataset(torch.utils.data.Dataset):
    def __init__(self, TTFS_path='/home/lymao/TTFS/ttfs.pkl'):
         super(TTFSDataset).__init__()
         with open(TTFS_path,"rb") as f:
            self.TTFS_data = pickle.load(f)
    
    def __getitem__(self, i):
         return self.TTFS_data[i]

    def __len__(self):
        return len(self.TTFS_data)

class TTFS():
    def __init__(self, step_num, device,  mnist_path='./mnist'):
        self.step_num = step_num
        self.mnist_path = mnist_path
        self.device = device

    def build_TTFS(self, save_path='/home/lymao/TTFS/ttfs.pkl'):
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
        mnist_train = datasets.MNIST(self.mnist_path, train=True, download=True, transform=transform)
        E = np.eye(self.step_num)
        TTFS_data = []
        for data, label in mnist_train:
            data = np.array(data)
            flatten_img = data.reshape([-1])
            flatten_img = (flatten_img * 24.5).astype(np.int)
            ttfs = E[flatten_img]
            TTFS_data.append(torch.tensor(ttfs.T))
        with open(save_path, "wb") as f:
            pickle.dump(TTFS_data, f)
        print("successfully saved TTFS data")

    def TTFS_to_image(self, ttfs, batches_done):
        intensity = torch.arange(0, self.step_num).to(device=self.device, dtype=torch.float) / 24
        intensity = (intensity @ ttfs) / torch.sum(ttfs, dim=0)
        img = intensity.reshape((1, 28, 28))
        save_image(img, "images/%d.png" % batches_done,)

# ttfs = TTFS(step_num=25, mnist_path='./mnist')
# ttfs.build_TTFS()