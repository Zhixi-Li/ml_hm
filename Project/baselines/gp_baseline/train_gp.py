"""
GP Baseline Training Script
使用监督学习训练GP模型求解TSP-50
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from ml4co_kit import TSPSolver, TSPEvaluator
from model import GPModel
from utils.dataset import TSPEnv, FakeDataset


########################
# Training Configuration
########################

# Paths
TRAIN_DATA_PATH = '../../data/train/tsp50_uniform_train_128k.txt'
VAL_DATA_PATH = '../../data/val/tsp50_uniform_val_128.txt'
SAVE_DIR = './checkpoints'

# Model parameters
NUM_LAYERS = 6
HIDDEN_DIM = 128
AGGREGATION = 'sum'

# Training parameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5

# Device
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Validation
VAL_INTERVAL = 1  # Validate every N epochs


def train_epoch(model, env, num_batches, batch_size, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch:3d}/{total_epochs}", unit="batch", leave=False)
    for _ in pbar:
        # Generate batch
        x, graph, ground_truth, nodes_num_list = env.generate_train_data(batch_size)
        x = x.to(device)
        graph = graph.to(device)
        ground_truth = ground_truth.to(device).long()

        optimizer.zero_grad()

        # Forward pass: 2-channel logits
        logits = model.encoder(x)  # (B, V, V, 2)
        B, V, _, _ = logits.shape

        # Cross-entropy loss
        loss = nn.functional.cross_entropy(logits.reshape(B * V * V, 2), ground_truth.reshape(B * V * V))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


def validate(model, env, device):
    """
    Validate the model using ML4CO-kit
    Args:
        model: GNN model
        env: TSPEnv with validation data
        device: computation device
    Returns:
        avg_cost: average tour cost
    """
    model.eval()

    # Get validation data
    x, graph, ground_truth, nodes_num_list = env.generate_val_data()
    num_instances = x.shape[0]

    total_cost = 0.0
    with torch.no_grad():
        for i in range(num_instances):
            coords = x[i:i+1].to(device)  # (1, V, 2)
            logits = model.encoder(coords)  # (1, V, V, 2)
            tours = model.decoder.decode(logits, coords)
            tour = tours[0]  # (V+1,)

            # Evaluate using ML4CO-kit
            evaluator = TSPEvaluator(x[i].cpu().numpy())
            cost = evaluator.evaluate(tour)
            total_cost += cost

    avg_cost = total_cost / num_instances
    return avg_cost


def main():
    print("=" * 60)
    print("GP Baseline Training - TSP-50")
    print("=" * 60)

    # Create timestamped run directory under checkpoints/
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(SAVE_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Checkpoint dir:    {run_dir}")

    # Create data environment
    print(f"\nInitializing data environment...")
    env = TSPEnv(
        train_path=TRAIN_DATA_PATH,
        val_path=VAL_DATA_PATH,
        cache_dir='./cache'
    )
    train_size = env.get_train_size()
    val_size = env.get_val_size()
    print(f"Training instances: {train_size}")
    print(f"Validation instances: {val_size}")

    # Calculate batches per epoch
    num_batches_per_epoch = train_size // BATCH_SIZE

    # Create model
    print(f"\nCreating model...")
    model = GPModel(
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        aggregation=AGGREGATION
    )
    model = model.to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    print(f"  - Num layers: {NUM_LAYERS}")
    print(f"  - Hidden dim: {HIDDEN_DIM}")
    print(f"  - Aggregation: {AGGREGATION}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Training loop
    print(f"\nStarting training...")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Batches per epoch: {num_batches_per_epoch}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Device: {DEVICE}")
    print("=" * 60)

    best_val_cost = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, env, num_batches_per_epoch, BATCH_SIZE, optimizer, DEVICE, epoch, EPOCHS)

        print(f"Epoch {epoch:3d}/{EPOCHS} - Train Loss: {train_loss:.4f}", end='')

        # Validate
        if epoch % VAL_INTERVAL == 0:
            val_cost = validate(model, env, DEVICE)
            print(f" | Val Cost: {val_cost:.4f}", end='')

            # Save best model
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                save_path = os.path.join(run_dir, 'best_model.pth')
                torch.save(model.state_dict(), save_path)
                save_path = os.path.join(run_dir, f'model_epoch_{epoch}_cost_{val_cost:.4f}.pth')
                torch.save(model.state_dict(), save_path)
                print(f" | *** New best model saved ***", end='')

        print()  # New line

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_path = os.path.join(run_dir, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)

    print("=" * 60)
    print(f"Training completed!")
    print(f"Best validation cost: {best_val_cost:.4f}")
    print(f"Model saved to: {run_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
