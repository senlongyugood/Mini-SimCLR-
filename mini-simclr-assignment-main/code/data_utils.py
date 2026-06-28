import torch
import torchvision
import torchvision.transforms as transforms


class SimCLRTransform:
    """Return two independent augmented views of one CIFAR-10 image."""

    def __init__(self, size=32):
        color_jitter = transforms.ColorJitter(
            brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
        )

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def _limit_dataset(dataset, subset_size):
    if subset_size is None:
        return dataset

    subset_size = min(int(subset_size), len(dataset.data))
    dataset.data = dataset.data[:subset_size]
    dataset.targets = dataset.targets[:subset_size]
    return dataset


def get_cifar10_dataloader(data_dir, batch_size=64, num_workers=0, train_size=None,
                           download=True):
    """CIFAR-10 train loader for SimCLR pretraining."""
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=SimCLRTransform(size=32),
    )
    dataset = _limit_dataset(dataset, train_size)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )


def get_cifar10_linear_dataloader(data_dir, batch_size=64, train=True,
                                  num_workers=0, subset_size=None,
                                  download=True):
    """CIFAR-10 loader with labels for linear probe training/evaluation."""
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=download,
        transform=transform,
    )
    dataset = _limit_dataset(dataset, subset_size)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=False,
    )


def get_cifar10_test_dataloader(data_dir, batch_size=64, num_workers=0,
                                test_size=None, download=True):
    return get_cifar10_linear_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        train=False,
        num_workers=num_workers,
        subset_size=test_size,
        download=download,
    )
