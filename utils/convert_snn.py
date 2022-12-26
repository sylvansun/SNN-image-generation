import colorsys

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop

import torch
import torch.nn as nn
import torch.nn.functional as F


class SNN_All_At_Once(nn.Module):
    """
    Covert a PyTorch model to an SNN model、
    Note:
        1. Not compatible if nn.Sequential is in model.modules().
        2. Insert a snn.Leaky layer after each nn.Module.
            - If you want to end the model without a snn.Leaky layer, set last_layer to the one you want.
        3. Only support LinearForward (e.g., output = fgh(x)).
            - If you want to use reslink, please inherit this class and rewrite the forward function.
    """

    def __init__(self, model: nn.Module, num_steps=25, beta=0.95, last_layer=None):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta
        self.nn_modules = []
        self.spiky_modules = []
        self.last_layer = nn.Identity() if last_layer is None else last_layer
        for i, (name, module) in enumerate(model.named_children()):
            self.nn_modules.append(name)
            setattr(self, name, module)
            setattr(self, f"lif{i}", snn.Leaky(beta=beta))
            self.spiky_modules.append(f"lif{i}")

    def forward(self, x):
        batch_size = x.shape[0]
        mems = [getattr(self, f"lif{i}").init_leaky() for i in range(len(self.nn_modules))]
        spk_rec = []
        mem_rec = []

        for step in range(self.num_steps):
            spk = x
            for i, name in enumerate(self.nn_modules):
                cur = getattr(self, name)(spk)
                spk, mem = getattr(self, f"lif{i}")(cur, mems[i])
                mems[i] = mem
            spk_rec.append(spk)
            mem_rec.append(self.last_layer(mem))

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)


if __name__ == "__main__":

    def yellow(s: str):
        return f"\033[33m{s}\033[0m"

    model = nn.Sequential(
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    print(yellow("Original model:"))
    print(model)
    snn_model = SNN_All_At_Once(model, num_steps=25, beta=0.95, last_layer=nn.Tanh())
    print(yellow("SNN model:"))
    print(snn_model)
    x = torch.randn(1, 28 * 28)
    print(yellow("Original model output:"))
    print(model(x))
    print(yellow("SNN model output:"))
    print(snn_model(x)[1].shape)
