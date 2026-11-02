from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_round_metric(history, metric_name: str, output_path: str | Path, title: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rounds = [entry.get('round', idx + 1) for idx, entry in enumerate(history)]
    values = [entry.get(metric_name, 0.0) for entry in history]
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, values, marker='o')
    plt.title(title)
    plt.xlabel('Communication round')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_client_distribution(label_map: dict[int, np.ndarray], output_path: str | Path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(label_map), 1, figsize=(9, 3 * max(1, len(label_map))))
    if len(label_map) == 1:
        axes = [axes]
    for client_id, labels in label_map.items():
        counts = np.bincount(labels.astype(int), minlength=10)
        axes[client_id].bar(np.arange(len(counts)), counts)
        axes[client_id].set_title(f'Client {client_id} class counts')
        axes[client_id].set_xlabel('Class')
        axes[client_id].set_ylabel('Samples')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
