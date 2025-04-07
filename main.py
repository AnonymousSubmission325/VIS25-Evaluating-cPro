# main.py

import os
import sys
import numpy as np
from src.evaluation.evaluation_runner import run_all_evaluations
from src.projections.circular_projection import circular_projection
from src.projections.circular_projection_adam import circular_projection_adam
from src.projections.circular_projection_pso import circular_projection_pso
from src.projections.circular_projection_lbfgs import circular_projection_lbfgs
# from src.projections.som_projection import sSOM
from src.projections.mds_projection import run_mds_with_time_limit
from src.projections.som_projection import som_projection
from src.projections.tow import spring_force_projection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
import umap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Define the projection configurations in main.py
PROJECTIONS_CONFIG = [
    # ("Simulated Annealing cPro", circular_projection, {"max_time": None}),
    ("Adam cPro", circular_projection_adam, {"max_time": None})
    # ("PSO cPro", circular_projection_pso, {"max_time": None}),
    # ("L-BFGS cPro", circular_projection_lbfgs, {"max_time": None}),
    # ("SOM", som_projection, {"show_plots": False, "labels": None, "max_time": None}),
    # ("Spring-Force Radial Projection", spring_force_projection, {"show_plots": False, "labels": None, "max_time": None}),
    # ("sSOM", sSOM, {"show_plots": False, "labels": None, "max_time": None}),
    # ("MDS 2D", run_mds_with_time_limit, {"n_components": 2, "max_time": None}),
    # ("MDS 1D", run_mds_with_time_limit, {"n_components": 1, "max_time": None}),
    # ("PCA 2D", lambda data: PCA(n_components=2).fit_transform(data), {}),
    # ("PCA 1D", lambda data: PCA(n_components=1).fit_transform(data), {}),
    # ("t-SNE 2D", lambda data: TSNE(n_components=2).fit_transform(data), {}),
    # ("UMAP 2D", lambda data: umap.UMAP(n_components=2).fit_transform(data), {}),
    # ("Isomap 2D", lambda data: Isomap(n_neighbors=5, n_components=2).fit_transform(data), {})

]

if __name__ == '__main__':
    # Set the flag to control plotting (True for showing plots, False to disable)
    show_plots = True
    # Run all evaluations with the configurations provided
    run_all_evaluations(show_plots=show_plots, max_time=1000, projections_config=PROJECTIONS_CONFIG)
