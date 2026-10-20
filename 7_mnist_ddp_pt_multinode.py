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


# Function to be run in each process
def init_dist():
    local_rank = int(os.environ["SLURM_LOCALID"])
    global_rank = int(os.environ['SLURM_PROCID'])
    node_id = int(os.environ["SLURM_NODEID"])
    world_size = int(os.environ['SLURM_NTASKS'])

    os.environ['RANK'] = str(global_rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group(backend='nccl', init_method='env://')

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # device = torch.cuda.current_device()

    print("running on rank {} (local: {}, node: {}) with world size {}".format(global_rank, local_rank, node_id, world_size))
    return global_rank, world_size, local_rank

def train_worker(epochs=5):
    global_rank, world_size, local_rank = init_dist()

    # Device
    device = torch.device(f"cuda:{local_rank}")

    # Dataset & loader (DistributedSampler)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=5000, sampler=sampler)

    # Model / optimizer
    model = Net().to(device)
    # PyTorch wrapper for DDP ** no need to manually sync gradients
    model = DDP(model)  # wrap in DDP
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001 * world_size)  # scale lr by world size

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

        print(f"[GPU {global_rank}] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # # Save curve per worker (only rank 0)
    # if rank == 0:
    #     plt.plot(losses)
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Training Loss")
    #     plt.title("MNIST DDP Training Loss Curve")
    #     os.makedirs("./img", exist_ok=True)
    #     plt.savefig("./img/mnist_loss_curve_ddp_pt.png")

    dist.destroy_process_group()


if __name__ == "__main__":
    # n_gpus = torch.cuda.device_count()
    print('Staring process...')
    
    start_time = time.time()
    train_worker(5)

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    if dist.is_initialized():
        dist.barrier()

    print('Program complete')

    if dist.is_initialized():
        dist.destroy_process_group()
