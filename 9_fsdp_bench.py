"""DDP vs ZeRO-1/2/3 on MNIST with torch.profiler + memory accounting.

Mode          Implementation
ddp           DDP + Adam                           (baseline, nothing sharded)
zero1         DDP + ZeroRedundancyOptimizer        (optimizer state sharded)
zero2         FSDP SHARD_GRAD_OP                   (opt state + grads sharded)
zero3         FSDP FULL_SHARD                      (opt state + grads + params sharded)

Launch (SLURM): 
    module load pytorch
    ./multinode_setup_and_run.sh 9_fsdp_bench.py
"""
import os, gc, time
from itertools import islice
import torch, torch.nn as nn, torch.optim as optim, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.profiler import profile, schedule, ProfilerActivity
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

HIDDEN, LAYERS, BATCH, STEPS = 4096, 4, 512, 8   # big enough that memory differences are visible
MB = 1024 ** 2


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [28 * 28] + [HIDDEN] * LAYERS
        blocks = []
        for i in range(LAYERS):
            blocks += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.fc = nn.Sequential(*blocks, nn.Linear(HIDDEN, 10))

    def forward(self, x):
        return self.fc(x)


def init_dist():
    if "SLURM_PROCID" in os.environ:
        rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
        local = int(os.environ["SLURM_LOCALID"])
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(local))
    else:  # torchrun
        rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", init_method="env://")
    return rank, world, local


def nbytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def opt_state_bytes(opt):
    inner = getattr(opt, "optim", opt)  # ZeroRedundancyOptimizer wraps the local optimizer in .optim
    return sum(nbytes([v for v in s.values() if torch.is_tensor(v)]) for s in inner.state.values())


def build(mode, local, lr):
    model = Net().cuda(local)
    if mode in ("ddp", "zero1"):
        model = DDP(model, device_ids=[local])
        if mode == "zero1":
            opt = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=optim.Adam, lr=lr)
        else:
            opt = optim.Adam(model.parameters(), lr=lr)
    else:
        strat = ShardingStrategy.SHARD_GRAD_OP if mode == "zero2" else ShardingStrategy.FULL_SHARD
        model = FSDP(model, sharding_strategy=strat, device_id=local,
                     auto_wrap_policy=ModuleWrapPolicy({nn.Linear}))  # one FSDP unit per Linear
        opt = optim.Adam(model.parameters(), lr=lr)
    return model, opt


def run(mode, rank, world, local, loader):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model, opt = build(mode, local, 1e-3)
    crit = nn.CrossEntropyLoss()
    params_mb = nbytes(p.data for p in model.parameters()) / MB  # local bytes (shards under FSDP)
    grads_mb = 0.0

    os.makedirs("./prof", exist_ok=True)
    sched = schedule(wait=2, warmup=2, active=4)
    t0 = None
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=sched,
                 profile_memory=True, record_shapes=False) as prof:
        for i, (x, y) in enumerate(islice(loader, STEPS)):
            if i == 4:
                torch.cuda.synchronize(); t0 = time.time()
            x, y = x.cuda(local, non_blocking=True), y.cuda(local, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            grads_mb = nbytes(p.grad for p in model.parameters()) / MB  # after backward, before step
            opt.step()
            prof.step()
    torch.cuda.synchronize()
    step_ms = (time.time() - t0) / (STEPS - 4) * 1000

    res = dict(mode=mode, params_MB=params_mb, grads_MB=grads_mb, opt_MB=opt_state_bytes(opt) / MB,
               peak_MB=torch.cuda.max_memory_allocated() / MB, step_ms=step_ms)
    if rank == 0:
        prof.export_chrome_trace(f"./prof/{mode}_rank0.json")
        print(f"\n=== {mode}: top CUDA kernels ===")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=8))
    del model, opt, prof
    return res


if __name__ == "__main__":
    rank, world, local = init_dist()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    if rank == 0:
        datasets.MNIST("./data", train=True, download=True)
    dist.barrier()
    ds = datasets.MNIST("./data", train=True, transform=tf)
    loader = DataLoader(ds, batch_size=BATCH, sampler=DistributedSampler(ds, world, rank, shuffle=True))

    results = []
    for mode in ["ddp", "zero1", "zero2", "zero3"]:
        results.append(run(mode, rank, world, local, loader))
        dist.barrier()

    if rank == 0:
        print(f"\nworld_size={world}  (MB per GPU, local view)")
        print(f"{'mode':<7}{'params':>10}{'grads':>10}{'opt state':>12}{'peak mem':>11}{'step ms':>10}")
        for r in results:
            print(f"{r['mode']:<7}{r['params_MB']:>10.1f}{r['grads_MB']:>10.1f}"
                  f"{r['opt_MB']:>12.1f}{r['peak_MB']:>11.1f}{r['step_ms']:>10.1f}")
    dist.destroy_process_group()