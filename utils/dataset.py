import torch
import numpy as np
import pickle
import os
from pathlib import Path
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snntorch.spikevision import spikedata


class TTFSDataset(torch.utils.data.Dataset):
    def __init__(self, TTFS_path=os.path.join(str(Path.cwd()), "asset/ttfs.pkl")):
        super(TTFSDataset).__init__()
        print("start building TTFS dataset")
        with open(TTFS_path, "rb") as f:
            self.TTFS_data = pickle.load(f)

    def __getitem__(self, i):
        return self.TTFS_data[i]

    def __len__(self):
        return len(self.TTFS_data)


class TTFS:
    def __init__(self, step_num, device, root="data"):
        self.step_num = step_num
        self.root = root
        self.device = device

    def build_TTFS(self, save_path=os.path.join(str(Path.cwd()), "asset/ttfs.pkl")):
        transform = get_transforms("mnist")
        mnist_train = datasets.MNIST(root=self.root, train=True, download=True, transform=transform)
        E = np.eye(self.step_num)
        TTFS_data = []
        for data, label in mnist_train:
            data = np.array(data)
            flatten_img = data.reshape([-1])
            flatten_img = (flatten_img * 24.5).astype(np.int64)
            ttfs = E[flatten_img]
            TTFS_data.append(torch.tensor(ttfs.T))
        with open(save_path, "wb") as f:
            pickle.dump(TTFS_data, f)
        print("successfully saved TTFS data")

    def TTFS_to_image(self, ttfs, batches_done):
        intensity = torch.arange(0, self.step_num).to(device=self.device, dtype=torch.float) / 24
        intensity = (intensity @ ttfs) / torch.sum(ttfs, dim=0)
        img = intensity.reshape((1, 28, 28))
        save_image(
            img,
            "images/ttfs_%d.png" % batches_done,
        )


def get_transforms(data_name):
    if data_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,)),
            ]
        )
    elif data_name == "cifar10":
        transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0,), (1,))]
        )
    else:
        raise Exception("Do not support transform for such a dataset")
    return transform


def get_dataset(batch_size, data_name):
    transform = get_transforms(data_name)
    if data_name == "mnist":
        train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    elif data_name == "cifar10":
        train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    else:
        raise Exception("Do not support such a dataset, please implement it yourself")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader


if __name__ == "__main__":
    ttfs = TTFS(25, "cuda")
    ttfs.build_TTFS()
