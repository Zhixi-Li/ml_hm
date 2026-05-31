"""
GP Baseline Evaluation Script
"""

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from ml4co_kit import TSPSolver, TSPEvaluator

# Add model directory to path
sys.path.insert(0, os.path.dirname(__file__))
from model import GPModel


########################
# Evaluation Configuration
########################

# Model configuration
MODEL_PATH = './checkpoints/best_model.pth'
NUM_LAYERS = 6
HIDDEN_DIM = 128
AGGREGATION = 'sum'

# Test data
TEST_DATA_PATH = '../../data/val/tsp50_uniform_val_128.txt'

# Device
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'


########################
# Model Configuration (for external evaluation)
########################

# Model parameters - students can modify these
model_params = {
    'num_layers': NUM_LAYERS,
    'hidden_dim': HIDDEN_DIM,
    'aggregation': AGGREGATION
}


def evaluate_model(model, test_solver, device):
    """
    Evaluate model using ML4CO-kit
    Args:
        model: trained GNN model
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
            # Solve
            coords = torch.from_numpy(test_points[i:i+1]).float().to(device)
            tours = model.solve(coords)
            tour = tours[0]

            # Evaluate using ML4CO-kit
            evaluator = TSPEvaluator(test_points[i])
            cost = evaluator.evaluate(tour)
            costs.append(cost)

            # Calculate gap if reference available
            if test_ref_tours is not None and len(test_ref_tours) > 0:
                ref_cost = evaluator.evaluate(test_ref_tours[i])
                ref_costs.append(ref_cost)
                gap = (cost - ref_cost) / ref_cost * 100
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
    print("GP Evaluation")
    print("=" * 60)

    # Load test data
    print(f"\nLoading test data from {TEST_DATA_PATH}...")
    test_solver = TSPSolver()
    test_solver.from_txt(
        TEST_DATA_PATH, ref=True, show_time=True, 
        normalize="uniform" not in TEST_DATA_PATH
    )
    print(f"Loaded {len(test_solver.points)} test instances")

    # Create model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = GPModel(**model_params)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {num_params:,} parameters")

    # Evaluate
    results = evaluate_model(model, test_solver, DEVICE)

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
        # print(f"Min gap:             {results['min_gap']:.2f}%")
        # print(f"Max gap:             {results['max_gap']:.2f}%")

    print(f"\nTiming:")
    print(f"Total time:          {results['total_time']:.2f}s")
    print(f"Avg time/instance:   {results['avg_time_per_instance']:.4f}s")

    print("=" * 60)


if __name__ == '__main__':
    main()
