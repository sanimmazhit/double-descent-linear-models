
# Double Descent in Linear Models

Individual project for Statistical Methods for Machine Learning
(A.Y. 2025/26).

## Project Overview

This project investigates the double descent phenomenon in
linear regression by examining how training and test errors
change as the number of features increases.

Least Squares and Ridge Regression are implemented from scratch
using NumPy.

## Experiments

1. Synthetic regression with sparse and dense signals
2. Double descent near the interpolation threshold
3. Comparison of Least Squares and Ridge Regression
4. Experiments repeated across random seeds
5. Effect of label noise
6. California Housing with Random Fourier Features
7. Validation-based Ridge regularization
8. Final comparison and results analysis

## Project Structure

- `src/` - Regression algorithms and data generation
- `notebooks/01_synthetic_experiments.ipynb` - Synthetic experiments
- `notebooks/02_real_world_experiments.ipynb` - California Housing experiments
- `notebooks/03_results_analysis.ipynb` - Final analysis
- `results/` - Experimental graphs and CSV files
- `requirements.txt` - Python dependencies

## Installation

Python 3.9 is recommended.

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install the required packages:

    python -m pip install -r requirements.txt

## Running the Experiments

Open the notebooks in VS Code or Jupyter.

Select the project's Python environment and run all cells
in the following order:

1. `01_synthetic_experiments.ipynb`
2. `02_real_world_experiments.ipynb`
3. `03_results_analysis.ipynb`

Results and visualizations are saved in the `results/` folder.

## Reference

Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019).
Reconciling modern machine-learning practice and the
classical bias-variance trade-off.
Proceedings of the National Academy of Sciences.

https://arxiv.org/abs/1812.11118
