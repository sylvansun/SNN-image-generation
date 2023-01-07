import matplotlib.pyplot as plt
from tqdm import tqdm

from snntorch import backprop
from snntorch import functional as SF

import torch
from models.snn_classifier import SNN
from utils.dataset import get_dataset

backprop.BPTT


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    train_loader, test_loader = get_dataset(batch_size=batch_size, data_name="cifar10")
    model = SNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = SF.ce_rate_loss()
    epoch = 20
    bar = tqdm(range(epoch), bar_format="{l_bar}{bar:20}{r_bar}", colour="#ffa500")
    accs = []
    for i in bar:
        model.train()
        for datas in train_loader:
            images, labels = datas
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            spk_rec, _ = model(images)
            loss = loss_fn(spk_rec, labels)
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
                spk_rec, _ = model(images)
                total += labels.size(0)
                correct += SF.accuracy_rate(spk_rec, labels) * labels.size(0)
            accuracy = correct / total
            accs.append(accuracy)
            bar.set_postfix_str("Accuracy: {:.2f}".format(accuracy))

    open("docs/acc_snn.txt", "w").write(str(accs))
    accs = [accs[0]] + [accs[i] * 0.7 + accs[i + 1] * 0.3 for i in range(len(accs) - 1)]
    plt.plot(accs)
    plt.savefig("docs/acc_snn.png")


if __name__ == "__main__":
    main()
