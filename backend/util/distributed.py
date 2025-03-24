import torch
import torch.distributed as dist

def setup(rank, world_size):
    """
    Initialize the distributed process group.
    """
    dist.init_process_group(
        backend='nccl',  # Use NCCL backend for GPU training
        init_method='env://',  # Use environment variables for initialization
        rank=rank,
        world_size=world_size
    )

def cleanup():
    """
    Clean up the distributed process group.
    """
    dist.destroy_process_group()