import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from hw2.models.cnn import CNN
from hw2.utils.dataset import get_dataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the dataset
    batch_size = 256
    train_loader, test_loader = get_dataset(batch_size)

    # Load the model
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epoch = 20
    bar = tqdm.tqdm(range(epoch), bar_format="{l_bar}{bar:20}{r_bar}", colour="#ffa500")
    accs = []
    for i in bar:
        model.train()
        for datas in train_loader:
            images, labels = datas
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for datas in test_loader:
                images, labels = datas
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            accs.append(accuracy)
            bar.set_postfix_str("Accuracy: {:.2f}".format(accuracy))

    open("docs/acc_cnn.txt", "w").write(str(accs))
    accs = [accs[0]] + [accs[i] * 0.2 + accs[i + 1] * 0.8 for i in range(len(accs) - 1)]
    plt.plot(accs)
    plt.savefig("docs/acc_cnn.png")


if __name__ == "__main__":
    main()
