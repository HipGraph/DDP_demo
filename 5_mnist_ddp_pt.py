import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import time

torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Simple Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(x)


def train_worker(rank, world_size, epochs=5):
    # init process group
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Dataset & loader (DistributedSampler)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=5000, sampler=sampler)

    # Model / optimizer
    model = Net().to(device)
    # PyTorch wrapper for DDP ** no need to manually sync gradients
    model = DDP(model)  # wrap in DDP
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    import time

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()


    losses = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # important for shuffle
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()      # DDP syncs gradients automatically
            optimizer.step()
            losses.append(loss.item())
        if epoch % 10 == 0 and rank == 0:
            print(f"[GPU {rank}] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Training time: {elapsed:.2f}s")

    # Save curve per worker (only rank 0)
    if rank == 0:
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("Training Loss")
        plt.title("MNIST DDP Training Loss Curve")
        os.makedirs("./img", exist_ok=True)
        plt.savefig("./img/mnist_loss_curve_ddp_pt.png")

    dist.destroy_process_group()


if __name__ == "__main__":
    # n_gpus = torch.cuda.device_count()
    print('Start MNIST training on multiple GPUs (DDP with PyTorch)...')
    world_size = 4
    start_time = time.time()
    mp.spawn(train_worker, args=(world_size, 5), nprocs=world_size, join=True)
    end_time = time.time()
    print(f"World Size: {world_size}. Training completed in {end_time - start_time:.2f} seconds.")
