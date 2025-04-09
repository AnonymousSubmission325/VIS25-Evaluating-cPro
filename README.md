# Projection Evaluation Framework

This repository provides a modular evaluation pipeline for comparing and analyzing high-dimensional data projection techniques, including custom circular projections and standard methods like PCA, t-SNE, and UMAP.

## Contents

- **scr/main.py**: Entry point for executing the complete evaluation pipeline.
- **src/**: Contains the core logic, dataset processing scripts, projection methods, evaluation utilities, and plotting routines.
- **results/**: Automatically generated results, including evaluation metrics and visualizations.

---

## How to Run

### Step 1: Install Requirements
Install the required dependencies in your Python environment:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Evaluation
From the project root, execute:
```bash
python main.py
```
This script:
- Loads selected real-world and artificial datasets.
- Applies projection techniques defined in `PROJECTIONS_CONFIG` in `main.py`.
- Evaluates each projection.
- Generates plots and saves results in the `results/` folder.

### Step 3: (Optional) Customize Datasets and Projections

#### Enable or disable datasets:
Edit `src/data/data_import.py` and uncomment the datasets you want to include in the `load_datasets()` or `load_artificial_datasets()` functions.

#### Enable or disable projections:
In `main.py`, adjust the `PROJECTIONS_CONFIG` list. Projections are defined as tuples:
```python
("Projection Name", function_reference, {optional_parameters})
```
Uncomment the lines to include standard or custom projection methods.

#### Adjust projection method parameters:
For each projection method, you can adjust internal settings directly in its respective implementation file under `src/projections/`. This allows full control over optimization routines, dimensionality settings, or distance functions.

---

## Directory Structure
```
project-root/
├── main.py
├── datasets/
│   └── preprocessed/           # Preprocessed real-world datasets (CSV)
├── results/
│   ├── *.png                   # Output visualizations
│   ├── records/                # Evaluation CSV results
│   └── plots/                  # Individual projection plots
├── src/
│   ├── data/                   # Data loaders and preprocessors
│   ├── evaluation/             # Evaluation and runners
│   ├── projections/            # Projection algorithm implementations
│   ├── utils/                  # Plotting and export helpers
│   └── ...
└── requirements.txt            # Dependency list
```

---

## Output
After running `main.py`, results are written to:

- `results/records/`: CSV files with recorded evaluation metrics
- `results/*.png`: Overview plots
- `results/plots/`: Per-dataset and per-method projection visualizations

---

## Notes
- All input datasets must include a `target` column for supervised evaluation.
- All projection methods must accept and return 2D arrays.
- The `max_time` parameter in `main.py` restricts the runtime per method.
- The three largest data files were sampled due to upload limits. The author will provide the complete data via the upload link by request. 

---

## License
This project is for academic and research purposes. For questions, contact the repository maintainer.

