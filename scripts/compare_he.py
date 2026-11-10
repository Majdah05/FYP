"""Run reproducibly paired plaintext/CKKS federated experiments and plot curves.

This harness uses the existing training loop, validation and held-out test, but
does not rerun the separate centralized baseline. No individual update reference
is retained here; precision comparisons belong to benchmark_ckks.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from main import PathMNISTSubsetFactory, evaluate_federated_test, initialize_global_model, set_seed
from federated_learning.config import parse_config
from federated_learning.datasets.partition import partition_dataset
from federated_learning.datasets.pathmnist_loader import load_pathmnist_data
from federated_learning.federated.server import run_federated_training
from federated_learning.utils.logging import save_json, save_metrics_csv


def run_pair(config, output_dir, data_dir='data'):
    output_dir = Path(output_dir)
    # Use unique parent directories to preserve previous experiment artifacts.
    output_dir.mkdir(parents=True, exist_ok=False)
    data = load_pathmnist_data(data_dir, batch_size=config.batch_size)
    indices = list(range(min(64, len(data['train_dataset'])))) if config.smoke_test else None
    factory = PathMNISTSubsetFactory(data['train_dataset'], config.batch_size, config.seed, indices)
    partitions = partition_dataset(factory.get_labels(), config.num_clients, config.distribution,
                                   config.dirichlet_alpha, config.seed)
    digest = hashlib.sha256(json.dumps(partitions).encode()).hexdigest()
    histories = {}
    for mode in ('plaintext', 'ckks'):
        set_seed(config.seed)
        effective = dataclasses.replace(config, aggregation=mode)
        folder = output_dir / mode
        model = initialize_global_model(effective.dataset, model_name=effective.model)
        model, history = run_federated_training(factory, data['val_loader'], effective, folder, model)
        test = evaluate_federated_test(model, data['test_loader'], effective, folder)
        save_json({'config': dataclasses.asdict(effective), 'partition_sha256': digest,
                   'history': history, 'test_metrics': test}, folder / 'federated_history.json')
        save_metrics_csv(history, folder / 'federated_history.csv')
        histories[mode] = history
        print(f'{mode} completed: validation accuracy={history[-1]["validation_accuracy"]:.6f}', flush=True)
    for plain, encrypted in zip(histories['plaintext'], histories['ckks']):
        if plain['selected_client_ids'] != encrypted['selected_client_ids']:
            raise RuntimeError('Paired experiments selected different clients.')
    save_json({'config': dataclasses.asdict(config), 'partition_sha256': digest,
               'same_seed_partition_and_client_schedule': True}, output_dir / 'pair_manifest.json')
    plot_pair(histories, output_dir)
    return histories


def plot_pair(histories, output_dir):
    for metric in ('validation_accuracy', 'validation_loss', 'round_time_seconds'):
        plt.figure(figsize=(7, 4))
        for mode, history in histories.items():
            plt.plot([r['round'] for r in history], [r[metric] for r in history], marker='o', label=mode)
        plt.xlabel('Communication round')
        plt.ylabel(metric.replace('_', ' '))
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric}.png')
        plt.close()
    plt.figure(figsize=(7, 4))
    for metric in ('he_encryption_time_seconds', 'he_decryption_time_seconds', 'he_serialization_time_seconds', 'he_aggregation_time_seconds'):
        history = histories['ckks']
        plt.plot([r['round'] for r in history], [r[metric] for r in history], marker='o', label=metric.removeprefix('he_').removesuffix('_time_seconds'))
    plt.xlabel('Communication round')
    plt.ylabel('Seconds')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'crypto_time.png')
    plt.close()
    plt.figure(figsize=(7, 4))
    for mode, key, label in (
        ('plaintext', 'plaintext_float_upload_bytes_estimate', 'Plaintext float storage estimate'),
        ('ckks', 'he_ciphertext_upload_bytes', 'Serialized ciphertext upload'),
    ):
        history = histories[mode]
        plt.plot([r['round'] for r in history], [r[key] for r in history], marker='o', label=label)
    plt.xlabel('Communication round')
    plt.ylabel('Upload payload (bytes)')
    plt.title('Simulated upload; metadata and downloads excluded')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'communication_upload.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='A new directory for this paired run.')
    parser.add_argument('--data_dir', default='data')
    args, experiment_args = parser.parse_known_args()
    config = parse_config(experiment_args)
    if config.dataset != 'pathmnist':
        parser.error('This dataset comparison harness supports PathMNIST.')
    run_pair(config, args.output, args.data_dir)


if __name__ == '__main__':
    main()
