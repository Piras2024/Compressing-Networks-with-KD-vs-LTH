import torch
from torch.utils.data import TensorDataset, DataLoader

from mnist1d.data import get_dataset, get_dataset_args


def load_mnist1d_tensors(add_channel_dim: bool = True):
    args = get_dataset_args()
    data = get_dataset(args=args)

    x_train = torch.tensor(data["x"], dtype=torch.float32)
    y_train = torch.tensor(data["y"], dtype=torch.long)

    x_test = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)

    if add_channel_dim:
        x_train = x_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)

    return x_train, y_train, x_test, y_test


def get_mnist1d_datasets(add_channel_dim: bool = True):
    x_train, y_train, x_test, y_test = load_mnist1d_tensors(
        add_channel_dim=add_channel_dim
    )

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    return train_ds, test_ds


def get_mnist1d_loaders(
    batch_size_train: int = 128,
    batch_size_test: int = 256,
    shuffle_train: bool = True,
    seed: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """
    Returns PyTorch DataLoaders for MNIST-1D.
    If seed is provided, training shuffling is deterministic.
    """
    train_ds, test_ds = get_mnist1d_datasets()

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=shuffle_train,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def get_dataset_info():
    args = get_dataset_args()
    data = get_dataset(args=args)

    return {
        "n_train": len(data["y"]),
        "n_test": len(data["y_test"]),
        "input_length": data["x"].shape[-1],
        "num_classes": len(data["templates"]["y"]),
    }
