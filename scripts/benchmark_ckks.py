"""Offline precision and client-count experiment on IDENTICAL local states.

Run as a module from the repository root: python -m scripts.benchmark_ckks
Plaintext references are deliberately confined to this offline harness.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from federated_learning.config import ExperimentConfig
from federated_learning.federated.aggregation import fedavg_aggregate
from federated_learning.federated.strategies import create_aggregation_strategy
from federated_learning.models.cnn import PathMNISTCNN
from federated_learning.utils.logging import save_json, save_metrics_csv


def benchmark(client_counts, repeats, full_model, output_dir, seed=42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    template = PathMNISTCNN().state_dict() if full_model else {
        'weights': torch.zeros(8201, dtype=torch.float64),
        'bias': torch.zeros(7, dtype=torch.float32), 'counter': torch.tensor(1), 'empty': torch.empty(0),
    }
    strategy = create_aggregation_strategy(ExperimentConfig(aggregation='ckks'), template)
    records, round_id = [], 0
    for count in client_counts:
        for repeat in range(repeats):
            round_id += 1
            states = []
            for _ in range(count):
                state = copy.deepcopy(template)
                for name, value in state.items():
                    if value.is_floating_point():
                        state[name] = torch.from_numpy(rng.normal(0, 0.1, tuple(value.shape))).to(value.dtype)
                states.append(state)
            counts = [20, 10, 30] + [5 + i for i in range(max(0, count - 3))]
            counts = counts[:count]
            start = time.perf_counter()
            plain = fedavg_aggregate(states, counts)
            plaintext_time = time.perf_counter() - start
            strategy.begin_round(round_id, tuple(enumerate(counts)))
            start = time.perf_counter()
            packaged = [strategy.client_encoder.prepare(s, n, i, round_id) for i, (s, n) in enumerate(zip(states, counts))]
            encrypted, metrics = strategy.aggregate([p[0] for p in packaged], counts, [p[1] for p in packaged])
            wall_time = time.perf_counter() - start
            ref = np.concatenate([value.double().numpy().reshape(-1) for value in plain.values() if value.is_floating_point()])
            actual = np.concatenate([value.double().numpy().reshape(-1) for value in encrypted.values() if value.is_floating_point()])
            error = np.abs(ref - actual)
            record = {
                'clients': count, 'repeat': repeat + 1, 'seed': seed, 'sample_counts': counts,
                'floating_values': len(ref), 'full_cnn_schema': full_model,
                'max_absolute_error': float(error.max()), 'rmse': float(np.sqrt(np.mean(error ** 2))),
                'max_relative_error_with_1e-6_floor': float(np.max(error / np.maximum(np.abs(ref), 1e-6))),
                'allclose_atol_1e-6_rtol_1e-6': bool(np.allclose(ref, actual, atol=1e-6, rtol=1e-6)),
                'plaintext_aggregation_time_seconds': plaintext_time,
                'ckks_pipeline_wall_time_seconds': wall_time,
                'plaintext_float_upload_bytes_estimate': count * sum(v.numel() * v.element_size() for v in template.values() if v.is_floating_point()),
                **metrics,
            }
            for dtype in (torch.float32, torch.float64):
                names = [name for name in plain if plain[name].dtype == dtype and plain[name].numel()]
                if names:
                    record[f'{str(dtype).removeprefix("torch.")}_max_absolute_error'] = max(
                        float((plain[name].double() - encrypted[name].double()).abs().max()) for name in names
                    )
            records.append(record)
            print(f"clients={count}, repeat={repeat+1}: max error={error.max():.3g}, HE={metrics['he_total_time_seconds']:.3f}s", flush=True)
    output_dir = Path(output_dir)
    save_json({'config': strategy.metadata, 'records': records}, output_dir / 'precision_benchmark.json')
    save_metrics_csv(records, output_dir / 'precision_benchmark.csv')
    for key, title, filename in (
        ('max_absolute_error', 'Offline CKKS weighted aggregation error', 'aggregation_error.png'),
        ('he_total_time_seconds', 'CKKS time vs participating clients', 'client_scaling.png'),
    ):
        plt.figure(figsize=(7, 4))
        for count in client_counts:
            values = [r[key] for r in records if r['clients'] == count]
            plt.errorbar(count, np.mean(values), yerr=np.std(values), fmt='o', capsize=4)
        plt.xlabel('Participating clients')
        plt.ylabel(key)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client_counts', type=int, nargs='+', default=[2, 3, 5])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--full_model', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='results/ckks_precision')
    args = parser.parse_args()
    if args.repeats < 1 or any(c < 1 for c in args.client_counts):
        parser.error('repeats and client counts must be positive')
    benchmark(args.client_counts, args.repeats, args.full_model, args.output_dir, args.seed)


if __name__ == '__main__':
    main()
