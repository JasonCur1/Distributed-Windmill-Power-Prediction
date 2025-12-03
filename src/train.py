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
from sklearn.metrics import r2_score

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

    with model.join(): # use join() to handle uneven batch counts
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

    avg_loss = total_loss / max(1, len(dataloader))
    epoch_time = time.time() - start_time

    loss_tensor = torch.tensor([avg_loss], dtype=torch.float32)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_avg_loss = loss_tensor.item() / world_size

    if rank == 0:
        print(f"Epoch {epoch} complete in {epoch_time:.2f}s")
        print(f"  Global average loss: {global_avg_loss:.6f}")

    return avg_loss

def validate(model, val_loader, criterion, rank, world_size):
    """Validate model on validation set and return advanced metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    inference_model = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = inference_model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    avg_loss = total_loss / max(1, len(val_loader))

    # Calculate R2 (Coefficient of Determination)
    val_r2 = r2_score(all_targets, all_preds)

    pred_std = all_preds.std()
    target_std = all_targets.std()

    loss_tensor = torch.tensor([avg_loss], dtype=torch.float32)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_val_loss = loss_tensor.item() / world_size

    metrics = {
        'val_loss': global_val_loss,
        'r2': val_r2,
        'pred_std': pred_std,
        'target_std': target_std
    }

    return metrics

def main():
    rank, world_size = setup_distributed()

    # Configuration
    BATCH_SIZE = 256
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001

    # Check for resumption
    start_epoch = 0
    resume_path = 'outputs/models/checkpoint_epoch_30.pt' # TODO: this can change depending on where we resume from

    if rank == 0 and not os.path.exists(resume_path):
        print("No checkpoint found, starting from scratch.")
        resume_path = None

    if rank == 0:
        print("\n" + "="*60)
        print("DISTRIBUTED WIND POWER PREDICTION TRAINING")
        print("="*60)
        print(f"Workers: {world_size}")
        print(f"Epochs: {NUM_EPOCHS}")
        if resume_path:
            print(f"RESUMING from {resume_path}")
        print("="*60 + "\n")

    # Load Data
    print(f"\n[Rank {rank}] Loading data...")
    train_loader = get_distributed_dataloader(rank, world_size, 'data/wind_turbine_train.csv', BATCH_SIZE)
    val_loader = get_distributed_dataloader(rank, world_size, 'data/wind_turbine_val.csv', BATCH_SIZE)

    # Create Model
    model = WindPowerModel()

    # Load state BEFORE wrapping in DDP if resuming
    optimizer_state = None
    if os.path.exists('outputs/models/checkpoint_epoch_30.pt'):
        print(f"[Rank {rank}] Loading checkpoint...")
        checkpoint = torch.load('outputs/models/checkpoint_epoch_30.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_state = checkpoint['optimizer_state_dict']
        start_epoch = checkpoint['epoch'] + 1

    model = DDP(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    criterion = torch.nn.HuberLoss()

    if rank == 0:
        os.makedirs('outputs/models', exist_ok=True)

    best_val_loss = float('inf')

    # Sync before starting
    dist.barrier()

    for epoch in range(start_epoch, NUM_EPOCHS):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
            print(f"{'='*60}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch, rank, world_size)

        val_metrics = validate(model, val_loader, criterion, rank, world_size)
        val_loss = val_metrics['val_loss']

        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(val_loss)

        if rank == 0:
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val R2:     {val_metrics['r2']:.4f}")
            print(f"  Pred Std:   {val_metrics['pred_std']:.4f} (Target Std: {val_metrics['target_std']:.4f})")
            print(f"  Current LR: {current_lr:.6f}")

            # Save if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'outputs/models/best_model.pt')
                print(f"  New best model saved!")

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, f'outputs/models/checkpoint_epoch_{epoch + 1}.pt')
                print(f"  Checkpoint saved.")

    if rank == 0:
        print("\nTRAINING COMPLETE")
        torch.save(model.module.state_dict(), 'outputs/models/final_model.pt')

    cleanup()

if __name__ == "__main__":
    main()