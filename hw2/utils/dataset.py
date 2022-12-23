from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0,), (1,))]
    )
    return transform


def get_dataset(batch_size):
    transform = get_transforms()
    train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
