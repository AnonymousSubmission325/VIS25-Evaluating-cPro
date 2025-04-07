import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial import distance_matrix
from scipy.optimize import dual_annealing
from pyswarm import pso
import torch
import time

def compute_hd_distances(points):
    """
    Compute high-dimensional distance matrix using cosine distance, normalized to [0, 1].
    """
    points_centered = points - np.mean(points, axis=0)
    hd_dist_mat = cosine_distances(points_centered) / 2
    return hd_dist_mat

def compute_ld_dist_matrix(ld_points, n):
    """
    Compute low-dimensional circular distance matrix.
    """
    dist_matrix = distance_matrix(ld_points, ld_points, p=1)
    return np.minimum(dist_matrix, 1 - dist_matrix)

def run_simulated_annealing_cpro_loss(points, maxiter=1000, max_time=None):
    points = np.array(points)
    n = points.shape[0]
    hd_dist_mat = compute_hd_distances(points)
    bounds = [(0, 1) for _ in range(n)]
    loss_records = []

    def loss(ld_points):
        x = ld_points.reshape((n, 1))
        ld_dist_mat = compute_ld_dist_matrix(x, n)
        return np.abs(hd_dist_mat - (2 * ld_dist_mat)).sum() / 2

    # Use a fixed initial guess for consistency
    initial_guess = np.random.uniform(0, 1, size=n)
    print(f"Initial loss for SA: {loss(initial_guess)}")  # Debug statement

    try:
        dual_annealing(loss, bounds=bounds, x0=initial_guess, callback=lambda x, f, _: loss_records.append(f), maxiter=maxiter)
    except TimeoutError:
        print("Optimization stopped due to time limit.")

    return loss_records


def run_pso_cpro_loss(points, maxiter=6000, swarmsize=3000, max_time=None, **kwargs):
    """
    Optimizes circular projection using particle swarm optimization (PSO).
    """
    points = np.array(points)
    n = points.shape[0]
    hd_dist_mat = compute_hd_distances(points)
    loss_records = []

    def loss(ld_points):
        x = ld_points.reshape((n, 1))
        ld_dist_mat = compute_ld_dist_matrix(x, n)
        raw_loss = np.abs(hd_dist_mat - (2 * ld_dist_mat)).sum()
        return raw_loss / (hd_dist_mat.size)  # Normalize by the number of elements


    lb = np.zeros(n)
    ub = np.ones(n)

    try:
        pso(loss, lb, ub, maxiter=maxiter, swarmsize=swarmsize, **kwargs)
    except Exception as e:
        print(e)

    return loss_records

def run_lbfgs_cpro_loss(points, lr=0.1, maxiter=100, max_time=None):
    """
    Optimizes circular projection using L-BFGS.
    """
    points = torch.tensor(points, dtype=torch.float32)
    n = points.shape[0]
    hd_dist_mat = torch.tensor(compute_hd_distances(points.numpy()), dtype=torch.float32)
    embedding = torch.randn(n, requires_grad=True)
    optimizer = torch.optim.LBFGS([embedding], lr=lr, max_iter=maxiter, line_search_fn='strong_wolfe')
    loss_records = []

    def compute_ld_dist_matrix_torch(ld_points):
        ld_points = ld_points.view(n, 1)
        dist_matrix = torch.cdist(ld_points, ld_points, p=1)
        return torch.minimum(dist_matrix, 1 - dist_matrix)

    def loss(ld_points):
        ld_dist_mat = compute_ld_dist_matrix_torch(ld_points)
        diff = torch.abs(hd_dist_mat - (2 * ld_dist_mat))
        return diff.sum() / 2

    start_time = time.time()
    def closure():
        optimizer.zero_grad()
        current_loss = loss(embedding)
        current_loss.backward()
        loss_records.append(current_loss.item())
        return current_loss

    for _ in range(maxiter):
        if max_time and (time.time() - start_time) > max_time:
            print("Optimization stopped due to time limit.")
            break
        optimizer.step(closure)

    return loss_records

def run_adam_cpro_loss(points, lr=0.1, maxiter=100, max_time=None):
    """
    Optimizes circular projection using Adam optimizer.
    """
    points = torch.tensor(points, dtype=torch.float32)
    n = points.shape[0]
    hd_dist_mat = torch.tensor(compute_hd_distances(points.numpy()), dtype=torch.float32)
    embedding = torch.randn(n, requires_grad=True)
    optimizer = torch.optim.Adam([embedding], lr=lr)
    loss_records = []

    def compute_ld_dist_matrix_torch(ld_points):
        ld_points = ld_points.view(n, 1)
        dist_matrix = torch.cdist(ld_points, ld_points, p=1)
        return torch.minimum(dist_matrix, 1 - dist_matrix)

    def loss(ld_points):
        ld_dist_mat = compute_ld_dist_matrix_torch(ld_points)
        diff = torch.abs(hd_dist_mat - (2 * ld_dist_mat))
        return diff.sum() / 2

    start_time = time.time()
    for _ in range(maxiter):
        if max_time and (time.time() - start_time) > max_time:
            print("Optimization stopped due to time limit.")
            break
        optimizer.zero_grad()
        current_loss = loss(embedding)
        current_loss.backward()
        optimizer.step()
        loss_records.append(current_loss.item())

    return loss_records
