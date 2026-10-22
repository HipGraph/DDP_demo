import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Function to be run in each process
def run(rank, world_size):
    # Initialize the process group
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    # if rank != 1:
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # Each process has a tensor with its rank + 1
    tensor = torch.tensor([rank+1.0])
    print(f"Rank {rank}, initial tensor: {tensor.item()}")
    # Perform all-reduce operation (sum) and compute mean
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    # Print the mean tensor from each process
    print(f"Rank {rank}, mean grad: {tensor.item()}")
    # Clean up
    dist.destroy_process_group()


def main():
    # Launch 4 processes for distributed training
    mp.spawn(run, args=(4,), nprocs=4, join=True)

if __name__ == "__main__":
    main()


    
