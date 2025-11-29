"""
All machines run this.
Rank 0 means coordinator node. All other ranks are workers.
Everyone trains together, and the backwards call aggregates weights
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DPP
import os

from model import WindPowerModel
from data_loader import get_distributed_loader

def setup_distributed():
    """Initialize distributed environment"""
    dist.init_process_group(
        backend='gloo', # use CPU
        init_method='env://',
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size() # How many machines we use total

    print(f"[Rank {rank}] Initialzied. World size: {world_size}")
    return rank, world_size

def cleanup():
    dist.destroy_process_group()

def train_one_epoch(model, dataloader, optimizer, criterion, epoch, rank):
    """Train for one epoch. Everyone does this"""
    model.train()
    total_loss = 0

    for batch_index, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward() # Gradients auto synced here
        optimizer.step()

        total_loss += loss.item()

        if rank == 0 and batch_index % 10 == 0: # For now only coordinator prints
            print(f"Epoch {epoch}, Batch {batch_index}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    rank, world_size = setup_distributed()

    train_loader = get_distributed_loader(
        rank=rank,
        world_size=world_size,
        batch_size=32 # TODO: maybe tune this
    )

    model = WindPowerModel()
    model = DPP(model) # DPP handles all inter machine comunication

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # TODO: maybe tune learning rate. This is standard though
    criterion = torch.nn.MSELoss()

    # Main training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch, rank)

        if rank == 0: # Only coordinator saves model data and checkpoints
            print(f"Epoch {epoch} complete. Avg loss: {avg_loss:.4f}")

            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(), # .module to get unwrapped model
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,

                }, f'outputs/models/checkpoint_epoch_{epoch}.pt')

    if rank == 0:
        torch.save(model.module.state_dict(), 'outputs/models/final_model.pt')
        print('Training complete.')

    cleanup()

if __name__ == "__main__":
    main()