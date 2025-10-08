import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Function to be run in each process
def init_dist():
    local_rank = int(os.environ["SLURM_LOCALID"])
    global_rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    node_rank = int(os.environ['SLURM_NODEID'])

    os.environ['RANK'] = str(global_rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group(backend='nccl', init_method='env://')

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    device = torch.cuda.current_device()

    print("running on rank {} (local: {}, node {}) with world size {}".format(global_rank, local_rank, node_rank, world_size))

    return global_rank, world_size, local_rank

def run():
    global_rank, world_size, local_rank = init_dist()
    
    tensor = torch.tensor([global_rank+1.0]).to(local_rank)
    print(f"Rank {global_rank}, initial tensor: {tensor}")
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    
    print(f"Rank {global_rank}, mean grad: {tensor.item()}")

# Entry point
def main():
    print('process spawned via srun!')
    
    run()
    
    if dist.is_initialized():
        dist.barrier()

    print('Program complete')

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

