"""
LC Baseline Evaluation Script
"""

import time
import torch
import numpy as np
from tqdm import tqdm
from ml4co_kit import TSPSolver, TSPEvaluator
from model import LCModel, TSPEnv


########################
# Evaluation Configuration
########################

# Model configuration
MODEL_PATH = 'checkpoints/best_model.pth'

# Problem parameters (change when testing different scales)
NODE_CNT = 50
POMO_SIZE = 50

# Model parameters
EMBEDDING_DIM = 128
NUM_ATT_LAYERS = 3
NUM_HEADS = 8
QKV_DIM = 16
FF_HIDDEN_DIM = 512
LOGIT_CLIPPING = 10

# Test data
TEST_DATA_PATH = '../../data/val/tsp50_uniform_val_128.txt'

# Device
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


########################
# Model Configuration (for external evaluation)
########################

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


def evaluate_model(model, env, test_solver, device):
    """
    Evaluate model using ML4CO-kit
    Args:
        model: trained POMO model
        env: TSP environment
        test_solver: TSPSolver with test data
        device: computation device
    Returns:
        results: dict with evaluation metrics
    """
    model.eval()

    test_points = test_solver.points
    test_ref_tours = test_solver.ref_tours
    num_instances = len(test_points)

    costs = []
    gaps = []
    ref_costs = []

    print(f"\nEvaluating on {num_instances} instances...")

    start_time = time.time()
    with torch.no_grad():
        for i in tqdm(range(num_instances), desc="Evaluating", unit="instance"):
            # Load problem
            coords = torch.from_numpy(test_points[i:i+1]).float().to(device)
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
            costs.append(tour_length)

            # Calculate gap if reference available
            if test_ref_tours is not None and len(test_ref_tours) > 0:
                evaluator = TSPEvaluator(test_points[i])
                ref_cost = evaluator.evaluate(test_ref_tours[i])
                ref_costs.append(ref_cost)
                gap = (tour_length - ref_cost) / ref_cost * 100
                gaps.append(gap)

    total_time = time.time() - start_time

    # Statistics
    results = {
        'num_instances': num_instances,
        'avg_cost': np.mean(costs),
        'std_cost': np.std(costs),
        'min_cost': np.min(costs),
        'max_cost': np.max(costs),
        'total_time': total_time,
        'avg_time_per_instance': total_time / num_instances,
    }

    if gaps:
        results['avg_gap'] = np.mean(gaps)
        results['std_gap'] = np.std(gaps)
        results['min_gap'] = np.min(gaps)
        results['max_gap'] = np.max(gaps)

    if ref_costs:
        results['avg_optimal_cost'] = np.mean(ref_costs)
        results['std_optimal_cost'] = np.std(ref_costs)
        results['min_optimal_cost'] = np.min(ref_costs)
        results['max_optimal_cost'] = np.max(ref_costs)

    return results


def main():
    print("=" * 60)
    print("LC Baseline Evaluation - TSP-50 with POMO")
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

    # Load test data
    print(f"\nLoading test data from {TEST_DATA_PATH}...")
    test_solver = TSPSolver()
    test_solver.from_txt(
        TEST_DATA_PATH, ref=True,
        normalize="uniform" not in TEST_DATA_PATH
    )
    print(f"Loaded {len(test_solver.points)} test instances")

    # Create model and environment
    print(f"\nLoading model from {MODEL_PATH}...")
    model = LCModel(**model_params)

    # Load weights - need to handle device mapping properly
    if 'cuda' in DEVICE:
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    env = TSPEnv(**env_params)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {num_params:,} parameters")

    # Evaluate
    results = evaluate_model(model, env, test_solver, DEVICE)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Number of instances: {results['num_instances']}")
    print(f"Average cost:        {results['avg_cost']:.4f}")
    # print(f"Std cost:            {results['std_cost']:.4f}")
    # print(f"Min cost:            {results['min_cost']:.4f}")
    # print(f"Max cost:            {results['max_cost']:.4f}")

    if 'avg_optimal_cost' in results:
        print(f"\nOptimal cost (reference):")
        print(f"Average optimal:     {results['avg_optimal_cost']:.4f}")
        # print(f"Std optimal:         {results['std_optimal_cost']:.4f}")
        # print(f"Min optimal:         {results['min_optimal_cost']:.4f}")
        # print(f"Max optimal:         {results['max_optimal_cost']:.4f}")

    if 'avg_gap' in results:
        print(f"\nGap to reference:")
        print(f"Average gap:         {results['avg_gap']:.2f}%")
        # print(f"Std gap:             {results['std_gap']:.2f}%")

    print(f"\nTiming:")
    print(f"Total time:          {results['total_time']:.2f}s")
    print(f"Avg time/instance:   {results['avg_time_per_instance']:.4f}s")

    print("=" * 60)


if __name__ == '__main__':
    main()
