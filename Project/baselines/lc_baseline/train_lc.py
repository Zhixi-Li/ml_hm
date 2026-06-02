"""
LC Baseline Training Script
使用POMO强化学习训练LC模型求解TSP-50
"""

import os
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from ml4co_kit import TSPSolver, TSPEvaluator
from model import LCModel, TSPEnv


########################
# Training Configuration
########################

# Paths
VAL_DATA_PATH = '../../data/val/tsp50_uniform_val_128.txt'
SAVE_DIR = './checkpoints'

# Problem parameters
NODE_CNT = 50
POMO_SIZE = 50

# Model parameters
EMBEDDING_DIM = 128
NUM_ATT_LAYERS = 3
NUM_HEADS = 8
QKV_DIM = 16
FF_HIDDEN_DIM = 512
LOGIT_CLIPPING = 10

# Training parameters
BATCH_SIZE = 64
EPOCHS = 100
BATCHES_PER_EPOCH = 100
LEARNING_RATE = 1e-4

# Validation
VAL_INTERVAL = 1

# Device
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train_one_batch(model, env, optimizer, batch_size):
    """Train on one batch using REINFORCE"""
    # Generate random instances
    env.load_problems(batch_size)

    # Reset environment
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    # Collect trajectories
    state, reward, done = env.pre_step()
    prob_list = []

    while not done:
        selected, prob = model(state)
        state, reward, done = env.step(selected)
        if prob is not None:
            prob_list.append(prob)

    # REINFORCE loss
    # reward shape: (batch, pomo)
    # Baseline: mean reward across POMO
    advantage = reward - reward.mean(dim=1, keepdim=True)

    # Log probability
    log_prob = torch.stack(prob_list, dim=2).log().sum(dim=2)  # (batch, pomo)

    # Loss
    loss = -(advantage * log_prob).mean()

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Return average tour length
    avg_length = -reward.mean().item()
    return avg_length, loss.item()


def validate(model, env, val_solver, device):
    """
    Validate using ML4CO-kit
    Args:
        model: POMO model
        env: TSP environment
        val_solver: TSPSolver with validation data
        device: computation device
    Returns:
        avg_cost: average tour cost
    """
    model.eval()

    val_points = val_solver.points
    num_instances = len(val_points)

    total_cost = 0.0

    with torch.no_grad():
        for i in range(num_instances):
            # Load problem
            coords = torch.from_numpy(val_points[i:i+1]).float().to(device)
            problems = torch.cdist(coords, coords, p=2)
            env.load_problems_manual(problems, coords)

            # Reset and solve
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                state, reward, done = env.step(selected)

            # Get best tour from POMO
            best_reward = reward.max().item()
            tour_length = -best_reward

            total_cost += tour_length

    model.train()
    avg_cost = total_cost / num_instances
    return avg_cost


def main():
    print("=" * 60)
    print("LC Baseline Training - TSP-50 with POMO")
    print("=" * 60)

    # Set CUDA device first
    if 'cuda' in DEVICE:
        if ':' in DEVICE:
            cuda_device_num = int(DEVICE.split(':')[1])
        else:
            cuda_device_num = 0
        torch.cuda.set_device(cuda_device_num)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Create timestamped run directory under checkpoints/
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(SAVE_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Checkpoint dir:    {run_dir}")

    # Model parameters
    model_params = {
        'embedding_dim': EMBEDDING_DIM,
        'sqrt_embedding_dim': EMBEDDING_DIM ** 0.5,
        'num_att_layers': NUM_ATT_LAYERS,
        'qkv_dim': QKV_DIM,
        'sqrt_qkv_dim': QKV_DIM ** 0.5,
        'num_heads': NUM_HEADS,
        'logit_clipping': LOGIT_CLIPPING,
        'ff_hidden_dim': FF_HIDDEN_DIM,
        'eval_type': 'argmax',
    }

    # Environment parameters
    env_params = {
        'task': 'TSP',
        'node_cnt': NODE_CNT,
        'pomo_size': POMO_SIZE
    }

    # Create model and environment
    print(f"\nCreating model and environment...")
    model = LCModel(**model_params)
    env = TSPEnv(**env_params)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    print(f"  - Embedding dim: {EMBEDDING_DIM}")
    print(f"  - Num attention layers: {NUM_ATT_LAYERS}")
    print(f"  - Num heads: {NUM_HEADS}")
    print(f"  - POMO size: {POMO_SIZE}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load validation data
    print(f"\nLoading validation data from {VAL_DATA_PATH}...")
    val_solver = TSPSolver()
    val_solver.from_txt(VAL_DATA_PATH, ref=True)
    print(f"Loaded {len(val_solver.points)} validation instances")

    # Training loop
    print(f"\nStarting training...")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batches per epoch: {BATCHES_PER_EPOCH}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Device: {DEVICE}")
    print("=" * 60)

    best_val_cost = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_length = 0.0
        epoch_loss = 0.0

        pbar = tqdm(range(BATCHES_PER_EPOCH), desc=f"Epoch {epoch:3d}/{EPOCHS}", unit="batch", leave=False)
        for _ in pbar:
            avg_length, loss = train_one_batch(model, env, optimizer, BATCH_SIZE)
            epoch_length += avg_length
            epoch_loss += loss
            pbar.set_postfix({"length": f"{avg_length:.4f}", "loss": f"{loss:.4f}"})

        epoch_length /= BATCHES_PER_EPOCH
        epoch_loss /= BATCHES_PER_EPOCH

        print(f"Epoch {epoch:3d}/{EPOCHS} - Train Length: {epoch_length:.4f}, Loss: {epoch_loss:.4f}", end='')

        # Validation
        if epoch % VAL_INTERVAL == 0:
            val_cost = validate(model, env, val_solver, DEVICE)
            print(f" | Val Cost: {val_cost:.4f}", end='')

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
