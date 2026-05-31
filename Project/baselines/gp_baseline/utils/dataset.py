"""
TSP Data Loading using TSPSolver + tsp_dense_process pattern
Adapted from GP/gnn4co/env/env.py and GP/gnn4co/env/denser.py
"""

import os
import torch
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from ml4co_kit import TSPSolver, to_tensor, check_dim
from torch.utils.data import Dataset


def tsp_dense_process(points: np.ndarray, ref_tour: np.ndarray):
    """
    Convert TSP instance to dense graph representation
    Args:
        points: (V, 2) node coordinates
        ref_tour: (V+1,) reference tour (0-indexed)
    Returns:
        x: (V, 2) node features
        graph: (V, V) distance matrix
        ground_truth: (V, V) binary edge matrix (symmetric)
        nodes_num: int
    """
    check_dim(points, 2)
    check_dim(ref_tour, 1)

    nodes_num = points.shape[0]

    # Node features and distance matrix
    x = to_tensor(points)
    graph = to_tensor(cdist(points, points)).float()

    # Ground truth edge matrix
    if ref_tour is not None:
        ground_truth = torch.zeros(size=(nodes_num, nodes_num))
        for idx in range(len(ref_tour) - 1):
            ground_truth[ref_tour[idx]][ref_tour[idx+1]] = 1
        ground_truth = ground_truth + ground_truth.T
    else:
        ground_truth = None

    return (
        x,  # (V, 2)
        graph.float(),  # (V, V)
        ground_truth.long(),  # (V, V)
        nodes_num,
    )


def tsp_batch_data_process(points_batch, ref_tours_batch):
    """
    Process a batch of TSP instances
    Args:
        points_batch: list of (V, 2) arrays
        ref_tours_batch: list of (V+1,) arrays
    Returns:
        x: (B, V, 2) batched node features
        graph: (B, V, V) batched distance matrices
        ground_truth: (B, V, V) batched edge labels
        nodes_num_list: list of V for each instance
    """
    batch_data = [tsp_dense_process(points, ref_tour)
                  for points, ref_tour in zip(points_batch, ref_tours_batch)]

    x_list = [item[0] for item in batch_data]
    graph_list = [item[1] for item in batch_data]
    ground_truth_list = [item[2] for item in batch_data]
    nodes_num_list = [item[3] for item in batch_data]

    # Stack into batches
    x = torch.stack(x_list, dim=0)  # (B, V, 2)
    graph = torch.stack(graph_list, dim=0)  # (B, V, V)
    ground_truth = torch.stack(ground_truth_list, dim=0)  # (B, V, V)

    return x, graph, ground_truth, nodes_num_list


class FakeDataset(Dataset):
    """Fake dataset for epoch iteration"""
    def __init__(self, data_size):
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return idx


class TSPEnv:
    """
    TSP data environment using TSPSolver + tsp_dense_process
    Adapted from GP/gnn4co/env/env.py
    """
    def __init__(self, train_path, val_path, cache_dir='./cache'):
        self.train_path = train_path
        self.val_path = val_path
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Cache for training data
        self.train_cache = None
        self.train_data_size = 0

        # Validation data (loaded once)
        self.val_solver = None

    def _get_train_cache_path(self):
        """Get cache file path for training data"""
        filename = os.path.basename(self.train_path).replace('.txt', '.pt')
        return os.path.join(self.cache_dir, filename)

    def _load_train_cache(self):
        """Load training data with caching"""
        cache_path = self._get_train_cache_path()

        if os.path.exists(cache_path):
            # Load from cache
            print(f"  Loading training cache from {cache_path}...")
            cache = torch.load(cache_path)
            self.train_cache = cache
            self.train_data_size = cache['data_size']
        else:
            # Load from txt with progress bar via ml4co-kit
            solver = TSPSolver()
            solver.from_txt(self.train_path, ref=True, show_time=True)

            self.train_cache = {
                'points': solver.points,  # (N, V, 2)
                'ref_tours': solver.ref_tours,  # (N, V+1)
                'data_size': len(solver.points)
            }
            self.train_data_size = len(solver.points)

            # Save cache
            torch.save(self.train_cache, cache_path)
            print(f"  Training data cached to {cache_path}")

    def generate_train_data(self, batch_size):
        """
        Generate training data batch
        Args:
            batch_size: number of instances per batch
        Returns:
            x: (B, V, 2)
            graph: (B, V, V)
            ground_truth: (B, V, V)
            nodes_num_list: list of int
        """
        if self.train_cache is None:
            self._load_train_cache()

        # Random sample batch_size instances
        indices = np.random.choice(self.train_data_size, size=batch_size, replace=False)

        points_batch = [self.train_cache['points'][i] for i in indices]
        ref_tours_batch = [self.train_cache['ref_tours'][i] for i in indices]

        return tsp_batch_data_process(points_batch, ref_tours_batch)

    def generate_val_data(self):
        """
        Generate validation data (all instances)
        Returns:
            x: (N, V, 2)
            graph: (N, V, V)
            ground_truth: (N, V, V)
            nodes_num_list: list of int
        """
        if self.val_solver is None:
            self.val_solver = TSPSolver()
            self.val_solver.from_txt(self.val_path, ref=True, show_time=True)

        points_batch = list(self.val_solver.points)
        ref_tours_batch = list(self.val_solver.ref_tours)

        return tsp_batch_data_process(points_batch, ref_tours_batch)

    def get_train_size(self):
        """Get training data size"""
        if self.train_cache is None:
            self._load_train_cache()
        return self.train_data_size

    def get_val_size(self):
        """Get validation data size"""
        if self.val_solver is None:
            self.val_solver = TSPSolver()
            self.val_solver.from_txt(self.val_path, ref=True, show_time=True)
        return len(self.val_solver.points)
