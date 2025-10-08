# train_ddp_mnist_h5.py
import os
import time
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler


# ---------------------------
# 1. Simple Model
# ---------------------------
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



# ---------------------------
# 2. HDF5-backed Dataset
# ---------------------------
class MNISTH5Dataset(Dataset):
    def __init__(self, path="mnist_grouped.h5", train=True):
        self.path = path
        self.group = "train" if train else "test"
        self.f = h5py.File(self.path, "r", swmr=True, libver="latest")
        self.images = self.f[f"{self.group}/images"]  # (N_groups, 1000, 1, 28, 28)
        self.labels = self.f[f"{self.group}/labels"]  # (N_groups, 1000)
        # print(f"[Info] Opened {self.path} with {len(self.images)} batches of 1000 samples each.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imgs = torch.tensor(self.images[idx], dtype=torch.float32)   # (100, 1, 28, 28)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)    # (100,)
        return imgs.view(imgs.shape[0], -1), labels

def collate_batches(batch):
    # batch: list of (imgs, labels)
    imgs = torch.cat([b[0] for b in batch], dim=0)      # (batch_size*100, 784)
    labels = torch.cat([b[1] for b in batch], dim=0)
    return imgs, labels

# ---------------------------
# 3. Distributed init
# ---------------------------
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

    print("running on rank {} (local: {}, node: {}) with world size {}".format(global_rank, local_rank, node_id, world_size), flush=True)
    return global_rank, world_size, local_rank


# ---------------------------
# 4. Training Function
# ---------------------------
def train_worker(epochs=5, h5_path="mnist_grouped.h5"):
    
    global_rank, world_size, local_rank = init_dist()

    
    # Device
    device = torch.device(f"cuda:{local_rank}")
    dataset = MNISTH5Dataset(path=h5_path, train=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, collate_fn=collate_batches, num_workers=0, pin_memory=True)

    # Model / optimizer
    model = Net().to(device)
    # PyTorch wrapper for DDP ** no need to manually sync gradients
    model = DDP(model, device_ids=[local_rank])  # wrap in DDP
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001 * world_size)  # scale lr by world size

    losses = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # important for shuffle
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()      # DDP syncs gradients automatically
            optimizer.step()
            losses.append(loss.item())

        print(f"[GPU {global_rank}] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}", flush=True)

    # # Save curve per worker (only rank 0)
    # if rank == 0:
    #     plt.plot(losses)
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Training Loss")
    #     plt.title("MNIST DDP Training Loss Curve")
    #     os.makedirs("./img", exist_ok=True)
    #     plt.savefig("./img/mnist_loss_curve_ddp_pt.png")

    dist.destroy_process_group()


# ---------------------------
# 5. Entry Point
# ---------------------------

if __name__ == "__main__":
    # n_gpus = torch.cuda.device_count()
    print('Staring process...', flush=True)
    
    start_time = time.time()
    train_worker()

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.", flush=True)
    if dist.is_initialized():
        dist.barrier()

    print('Program complete', flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()

