import torch
import torchvision
import torchvision.transforms as transforms


class SimCLRTransform:
    """对同一张 CIFAR-10 图像生成两种随机增强视图"""

    def __init__(self, size=32):
        # 颜色抖动参数
        color_jitter = transforms.ColorJitter(
            brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
        )

        # 单视图增强流水线
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        """返回同一张图的两个增强视图"""
        return self.transform(x), self.transform(x)


def get_cifar10_dataloader(data_dir, batch_size=64, num_workers=0, train_size=None):
    """
    加载 CIFAR-10 训练集（不使用标签）

    参数:
        data_dir: CIFAR-10 数据存放路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        train_size: 只取前 N 张图（用于 CPU 快速实验），None 表示全量 50000
    """
    transform = SimCLRTransform(size=32)

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=transform
    )

    if train_size is not None:
        dataset.data = dataset.data[:train_size]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )
    return loader


def get_cifar10_test_dataloader(data_dir, batch_size=64, num_workers=0):
    """加载 CIFAR-10 测试集（线性评估用，保留标签）"""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=test_transform
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    return loader
