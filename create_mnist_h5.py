import os
import h5py
import torch
from torchvision import datasets, transforms


def create_mnist_h5(out_path="mnist_grouped.h5", group_size=1000):
    if os.path.exists(out_path):
        print(f"[Info] {out_path} already exists. Skipping.")
        return

    print("[Info] Downloading and writing MNIST to HDF5...")
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Convert to tensors
    train_imgs = torch.stack([img for img, _ in mnist_train])   # (60000, 1, 28, 28)
    train_labels = torch.tensor([label for _, label in mnist_train])
    test_imgs = torch.stack([img for img, _ in mnist_test])     # (10000, 1, 28, 28)
    test_labels = torch.tensor([label for _, label in mnist_test])

    def make_batches(x, size):
        n = x.shape[0]
        remainder = n % size
        if remainder:
            pad = size - remainder
            if x.ndim == 1:
                x = torch.cat([x, torch.zeros(pad, dtype=x.dtype)], dim=0)
            else:
                x = torch.cat([x, torch.zeros(pad, *x.shape[1:], dtype=x.dtype)], dim=0)
        return x.view(-1, size, *x.shape[1:])

    train_imgs_batched = make_batches(train_imgs, group_size)
    train_labels_batched = make_batches(train_labels, group_size)
    test_imgs_batched = make_batches(test_imgs, group_size)
    test_labels_batched = make_batches(test_labels, group_size)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("train/images", data=train_imgs_batched, compression="gzip")
        f.create_dataset("train/labels", data=train_labels_batched, compression="gzip")
        f.create_dataset("test/images", data=test_imgs_batched, compression="gzip")
        f.create_dataset("test/labels", data=test_labels_batched, compression="gzip")

    print(f"[Info] HDF5 dataset written to {out_path}")
    print(f"Train set: {train_imgs_batched.shape}, Test set: {test_imgs_batched.shape}")


if __name__ == "__main__":
    create_mnist_h5("mnist_grouped.h5", group_size=100)
