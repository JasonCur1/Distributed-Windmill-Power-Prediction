"""
All machines run this.
Rank 0 means coordinator node. All other ranks are workers.
Everyone trains together, and the backwards call aggregates weights
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
from datetime import datetime

from model import WindPowerModel
from data_loader import get_distributed_dataloader

def setup_distributed():
    """Initialize distributed environment"""
    dist.init_process_group(
        backend='gloo', # use CPU
        init_method='env://',
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size() # How many machines we use total

    print(f"[Rank {rank}] Initialized distributed training")
    print(f"[Rank {rank}] World size: {world_size}")

    return rank, world_size

def cleanup():
    dist.destroy_process_group()

def train_one_epoch(model, dataloader, optimizer, criterion, epoch, rank, world_size):
    """Train for one epoch. Everyone does this"""
    model.train()
    total_loss = 0
    num_batches = 0

    start_time = time.time()

    for batch_index, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward() # Gradients auto synced here
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if rank == 0 and batch_index % 100 == 0: # for now only coordinator prints
            print(f"  Batch {batch_index}/{len(dataloader)}, Loss: {loss.item():.6f}")

    avg_loss = total_loss / len(dataloader)
    epoch_time = time.time() - start_time

    loss_tensor = torch.tensor([avg_loss], dtype=torch.float32)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_avg_loss = loss_tensor.item() / world_size

    if rank == 0:
        print(f"Epoch {epoch} complete in {epoch_time:.2f}s")
        print(f"  Global average loss: {global_avg_loss:.6f}")

    return avg_loss

def validate(model, val_loader, criterion, rank):
    """Validate model on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    # Aggregate across workers
    loss_tensor = torch.tensor([avg_loss], dtype=torch.float32)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_val_loss = loss_tensor.item() / dist.get_world_size()

    return global_val_loss

def main():
    rank, world_size = setup_distributed()

    # Configuration
    BATCH_SIZE = 256
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001

    if rank == 0:
        print("\n" + "="*60)
        print("DISTRIBUTED WIND POWER PREDICTION TRAINING")
        print("="*60)
        print(f"Workers: {world_size}")
        print(f"Batch size per worker: {BATCH_SIZE}")
        print(f"Global batch size: {BATCH_SIZE * world_size}")
        print(f"Epochs: {NUM_EPOCHS}")
        print(f"Learning rate: {LEARNING_RATE}")
        print("="*60 + "\n")

    print(f"\n[Rank {rank}] Loading data...")
    train_loader = get_distributed_dataloader(
        rank=rank,
        world_size=world_size,
        data_path='data/wind_turbine_train.csv',
        batch_size=BATCH_SIZE,
        num_workers_dataloader=4
    )

    print(f"\n[Rank {rank}] Loading validation data...")
    val_loader = get_distributed_dataloader(
        rank=rank,
        world_size=world_size,
        data_path='data/wind_turbine_val.csv',
        batch_size=BATCH_SIZE,
        num_workers_dataloader=4
    )

    print(f"[Rank {rank}] Creating model...")
    model = WindPowerModel()
    model = DDP(model) # DDP handles all inter machine comunication

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print()

    if rank == 0:
        os.makedirs('outputs/models', exist_ok=True)
        os.makedirs('outputs/logs', exist_ok=True)

    best_val_loss = float('inf')

    dist.barrier() # sync barrier

    if rank == 0:
        print("Starting training...\n")

    # Training loop
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
            print(f"{'='*60}")

        # Train for one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch, rank, world_size
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, rank, world_size)

        if rank == 0:
            print(f"  Validation loss: {val_loss:.6f}")

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = 'outputs/models/best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, best_model_path)
                print(f"  New best model saved! (val_loss: {val_loss:.6f})")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f'outputs/models/checkpoint_epoch_{epoch + 1}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"  Checkpoint saved to {checkpoint_path}")

    # Training complete
    total_time = time.time() - start_time

    if rank == 0:
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Total training time: {total_time/60:.2f} minutes")
        print(f"Average time per epoch: {total_time/NUM_EPOCHS:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.6f}")

        final_model_path = 'outputs/models/final_model.pt'
        torch.save(model.module.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        print(f"Best model saved to outputs/models/best_model.pt")
        print("="*60 + "\n")

    cleanup()


if __name__ == "__main__":
    main()