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
VAL_DATASETS = {
    'tsp50_uniform': ('../../data/val/tsp50_uniform_val_128.txt', 0.6),
    'tsp50_ood': ('../../data/val/tsp50_ood_val_16.txt', 0.2),
    'tsp100_uniform': ('../../data/val/tsp100_uniform_val_16.txt', 0.2),
}
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
WEIGHT_DECAY = 1e-6
GRAD_CLIP_NORM = 1.0
CURRICULUM_START_EPOCH = 60

# Validation
VAL_INTERVAL = 1

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def augment_coords_d4(coords):
    """Random D4 symmetry augmentation for unit-square TSP coordinates."""
    ops = torch.randint(0, 8, (coords.size(0),), device=coords.device)
    x = coords[..., 0]
    y = coords[..., 1]
    augmented = torch.empty_like(coords)

    candidates = (
        torch.stack((x, y), dim=-1),
        torch.stack((1 - x, y), dim=-1),
        torch.stack((x, 1 - y), dim=-1),
        torch.stack((1 - x, 1 - y), dim=-1),
        torch.stack((y, x), dim=-1),
        torch.stack((1 - y, x), dim=-1),
        torch.stack((y, 1 - x), dim=-1),
        torch.stack((1 - y, 1 - x), dim=-1),
    )
    for op, candidate in enumerate(candidates):
        mask = ops == op
        if mask.any():
            augmented[mask] = candidate[mask]
    return augmented


def sample_curriculum_node_cnt(epoch):
    if epoch < CURRICULUM_START_EPOCH:
        return 50
    return int(np.random.choice([50, 75, 100], p=[0.6, 0.2, 0.2]))


def batch_size_for_node_cnt(node_cnt):
    return max(8, BATCH_SIZE * NODE_CNT // node_cnt)


def train_one_batch(model, env, optimizer, batch_size, node_cnt):
    """Train on one batch using REINFORCE"""
    # Generate random instances
    env.set_problem_size(node_cnt)
    env.load_problems(batch_size)
    env.coordinates = augment_coords_d4(env.coordinates)
    env.problems = torch.cdist(env.coordinates, env.coordinates, p=2)

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
    log_prob = torch.stack(prob_list, dim=2).clamp_min(1e-12).log().sum(dim=2)  # (batch, pomo)

    # Loss
    loss = -(advantage * log_prob).mean()

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    # Return average tour length
    avg_length = -reward.mean().item()
    return avg_length, loss.item()


def validate_solver(model, env, val_solver, device):
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
    total_gap = 0.0
    gap_count = 0

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
            if val_solver.ref_tours is not None and len(val_solver.ref_tours) > 0:
                evaluator = TSPEvaluator(val_points[i])
                ref_cost = evaluator.evaluate(val_solver.ref_tours[i])
                total_gap += (tour_length - ref_cost) / ref_cost * 100
                gap_count += 1

    model.train()
    avg_cost = total_cost / num_instances
    avg_gap = total_gap / gap_count if gap_count > 0 else None
    return avg_cost, avg_gap


def validate_all(model, env, val_solvers, device):
    results = {}
    weighted_gap = 0.0
    for name, (solver, weight) in val_solvers.items():
        avg_cost, avg_gap = validate_solver(model, env, solver, device)
        results[name] = {'cost': avg_cost, 'gap': avg_gap, 'weight': weight}
        metric = avg_gap if avg_gap is not None else avg_cost
        weighted_gap += weight * metric
    return results, weighted_gap


def print_validation_results(results, weighted_score):
    parts = []
    for name, metrics in results.items():
        if metrics['gap'] is None:
            parts.append(f"{name}: cost {metrics['cost']:.4f}")
        else:
            parts.append(f"{name}: gap {metrics['gap']:.2f}%")
    print(f" | Weighted Val: {weighted_score:.4f} | " + " | ".join(parts), end='')


def main():
    print("=" * 60)
    print("LC Baseline Training - TSP-50 with POMO")
    print("=" * 60)

    if DEVICE == 'cuda':
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
        'distance_bias': True,
        'distance_logit_bias': True,
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * 0.1
    )

    # Load validation data
    print(f"\nLoading validation data...")
    val_solvers = {}
    for name, (path, weight) in VAL_DATASETS.items():
        solver = TSPSolver()
        solver.from_txt(path, ref=True, normalize="uniform" not in path)
        val_solvers[name] = (solver, weight)
        print(f"  - {name}: {len(solver.points)} instances, weight={weight}")

    # Training loop
    print(f"\nStarting training...")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batches per epoch: {BATCHES_PER_EPOCH}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Curriculum starts: epoch {CURRICULUM_START_EPOCH}")
    print(f"  - Device: {DEVICE}")
    print("=" * 60)

    best_val_score = float('inf')
    submit_best_path = os.path.join(SAVE_DIR, 'best_model.pth')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_length = 0.0
        epoch_loss = 0.0

        pbar = tqdm(range(BATCHES_PER_EPOCH), desc=f"Epoch {epoch:3d}/{EPOCHS}", unit="batch", leave=False)
        for _ in pbar:
            node_cnt = sample_curriculum_node_cnt(epoch)
            current_batch_size = batch_size_for_node_cnt(node_cnt)
            avg_length, loss = train_one_batch(model, env, optimizer, current_batch_size, node_cnt)
            epoch_length += avg_length
            epoch_loss += loss
            pbar.set_postfix({"n": node_cnt, "length": f"{avg_length:.4f}", "loss": f"{loss:.4f}"})

        epoch_length /= BATCHES_PER_EPOCH
        epoch_loss /= BATCHES_PER_EPOCH

        print(f"Epoch {epoch:3d}/{EPOCHS} - Train Length: {epoch_length:.4f}, Loss: {epoch_loss:.4f}", end='')

        # Validation
        if epoch % VAL_INTERVAL == 0:
            val_results, val_score = validate_all(model, env, val_solvers, DEVICE)
            print_validation_results(val_results, val_score)

            if val_score < best_val_score:
                best_val_score = val_score
                save_path = os.path.join(run_dir, 'best_model.pth')
                torch.save(model.state_dict(), save_path)
                torch.save(model.state_dict(), submit_best_path)
                save_path = os.path.join(run_dir, f'model_epoch_{epoch}_score_{val_score:.4f}.pth')
                torch.save(model.state_dict(), save_path)
                print(f" | *** New best model saved ***", end='')

        print()  # New line
        scheduler.step()

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_path = os.path.join(run_dir, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)

    print("=" * 60)
    print(f"Training completed!")
    print(f"Best validation score: {best_val_score:.4f}")
    print(f"Model saved to: {run_dir}")
    print(f"Submission checkpoint: {submit_best_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
