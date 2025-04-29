# test_ddp.py
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    print(f"Rank ** {rank}: CUDA available={torch.cuda.is_available()}")
    tensor = torch.tensor([rank], device=f'cuda:{rank % 4}')
    dist.all_reduce(tensor)
    print(f"Rank ** {rank}: {tensor}")

if __name__ == "__main__":
    main()