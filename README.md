
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

```text
double-descent-linear-models/
│
├── src/
│   ├── data_generation.py          # Synthetic data generation
│   └── linear_models.py            # Least Squares and Ridge Regression
│
├── notebooks/
│   ├── 01_synthetic_experiments.ipynb
│   ├── 02_real_world_experiments.ipynb
│   └── 03_results_analysis.ipynb
│
├── results/
│   ├── synthetic_double_descent.png
│   ├── synthetic_ridge_comparison.png
│   ├── synthetic_repeated_experiments.png
│   ├── synthetic_noise_effect.png
│   ├── california_double_descent.png
│   ├── california_regularization_comparison.png
│   ├── california_ridge_tuning.png
│   ├── final_dataset_comparison.png
│   ├── synthetic_results.csv
│   ├── real_world_results.csv
│   └── final_summary.csv
│
├── report/
│   ├── main.tex
│   └── main.pdf
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Folder Descriptions

- **`src/`** — Custom implementations of the regression algorithms and synthetic data generators.
- **`notebooks/`** — Three Jupyter notebooks covering synthetic experiments, real-world experiments, and final results analysis.
- **`results/`** — Generated figures and CSV files containing experimental results.
- **`report/`** — LaTeX source and compiled PDF of the academic report.
- **`requirements.txt`** — Python packages and versions required to reproduce the experiments.
- **`README.md`** — Project overview, setup instructions, and experiment documentation.


## Running the Experiments

Open the notebooks in VS Code or Jupyter.

Select the project's Python environment and run all cells
in the following order:

1. `01_synthetic_experiments.ipynb`
2. `02_real_world_experiments.ipynb`
3. `03_results_analysis.ipynb`

Results and visualizations are saved in the `results/` folder.

## Project Report

The academic report is available in the `report/` folder.

The report includes:
- Introduction and theoretical background
- Experimental methodology
- Synthetic and real-world results
- Comparison of Least Squares and Ridge Regression
- Discussion, limitations, and conclusion
- References and academic integrity declaration

The report is written in LaTeX, and the PDF version
can be opened directly from GitHub.


## Reference

Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019).
Reconciling modern machine-learning practice and the
classical bias-variance trade-off.
Proceedings of the National Academy of Sciences.

https://arxiv.org/abs/1812.11118
