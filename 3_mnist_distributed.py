import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Simple Feedforward Neural Network for MNIST
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

# Training function for each worker
def train_worker(rank, epochs=5):
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 -> 784
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
    
    # Assign device based on rank
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Model, Loss, Optimizer
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    import time

    total_time = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    # Training loop
    losses = []
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch % 10 == 0 and rank == 0:
            print(f"[GPU {rank}] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Training time: {elapsed:.2f}s")

    # Save loss curve per worker
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title(f"MNIST Loss Curve (GPU {rank})")
    plt.savefig(f"./img/mnist_loss_curve_gpu{rank}.png")



if __name__ == "__main__":
    print('Start MNIST training on multiple GPUs (non-DDP)...')
    # n_gpus = torch.cuda.device_count()
    world_size = 4  # use up to 4 GPUs
    mp.spawn(train_worker, args=(5,), nprocs=world_size, join=True)
