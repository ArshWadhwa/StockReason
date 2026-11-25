"""Evaluation Result Visualizations Module."""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_return_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "results/figures/return_distribution.png",
):
    """Plots true vs predicted return distributions."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_true, bins=50, alpha=0.5, label="True Returns", density=True)
    ax.hist(y_pred, bins=50, alpha=0.5, label="Predicted Returns", density=True)
    ax.set_title("Return Distribution Comparison")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_error_distribution(
    errors: np.ndarray,
    output_path: str = "results/figures/error_distribution.png",
):
    """Plots forecast error distribution."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=50, color="crimson", alpha=0.7)
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Error (INR)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
