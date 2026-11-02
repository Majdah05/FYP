from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_experiment_summary(folder):
    summary_path = Path(folder) / 'comparison_summary.json'
    if not summary_path.exists():
        return {}
    with summary_path.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def main():
    results_dir = Path('results')
    experiment_dirs = [p for p in results_dir.iterdir() if p.is_dir()] if results_dir.exists() else []
    if not experiment_dirs:
        print('No experiment folders found in results/. Run at least one experiment first.')
        return

    summaries = [load_experiment_summary(p) for p in experiment_dirs]
    plt.figure(figsize=(8, 5))
    labels = []
    values = []
    for exp_dir, summary in zip(experiment_dirs, summaries):
        if summary:
            validation_metrics = summary.get('federated_validation', summary.get('federated', {}))
            if 'validation_accuracy' in validation_metrics:
                labels.append(exp_dir.name)
                values.append(validation_metrics['validation_accuracy'])
    plt.bar(labels, values)
    plt.title('Validation Accuracy Across Experiment Runs')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(results_dir / 'comparison_bar_chart.png')
    print(f'Comparison chart saved to {results_dir / "comparison_bar_chart.png"}')


if __name__ == '__main__':
    main()
