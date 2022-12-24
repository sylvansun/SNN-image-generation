from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(data_name):
    if data_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])
    elif data_name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
            ])
    else:
        raise Exception("Do not support transform for such a dataset")
    return transform


def get_dataset(batch_size, data_name):
    transform = get_transforms(data_name)
    if data_name == "mnist":
        train_dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="../data", train=False, download=True, transform=transform)
    elif data_name == "cifar10":
        train_dataset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)
    else:
        raise Exception("Do not support such a dataset, please implement it yourself")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataset(128, "cifar10")

    for i, (images, labels) in enumerate(train_loader):
        print(images.shape)
        print(labels)
        break