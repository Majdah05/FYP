from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from federated_learning.config import parse_config
from federated_learning.datasets.pathmnist_loader import load_pathmnist_data
from federated_learning.federated.server import initialize_global_model, run_federated_training
from federated_learning.training.centralized import CentralizedTrainer
from federated_learning.training.evaluation import evaluate_model
from federated_learning.utils.logging import save_json, save_metrics_csv
from federated_learning.utils.plotting import plot_round_metric


class PathMNISTSubsetFactory:
    def __init__(self, dataset, batch_size: int, seed: int = 42, sample_indices=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.sample_indices = list(sample_indices) if sample_indices is not None else list(range(len(dataset)))

    def get_labels(self):
        labels = np.asarray(self.dataset.labels).reshape(-1).astype(int)
        return labels[self.sample_indices]

    def create_subset(self, indices, shuffle: bool = False):
        source_indices = [self.sample_indices[index] for index in indices]
        subset = Subset(self.dataset, source_indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=shuffle)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_centralized_baseline(config, data_bundle, output_dir: Path):
    model = initialize_global_model('pathmnist', model_name=config.model, num_classes=9)
    device = config.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = CentralizedTrainer(
        model=model,
        device=device,
        optimizer_name=config.optimizer,
        lr=config.lr,
        momentum=config.momentum,
    )
    epochs = centralized_epoch_budget(config)
    result = trainer.train(data_bundle['train_loader'], data_bundle['val_loader'], epochs=epochs)
    test_metrics = evaluate_model(result['model'], data_bundle['test_loader'], device=str(trainer.device))
    test_metrics['dataset'] = config.dataset
    save_json({
        'config': config.__dict__,
        'test_metrics': test_metrics,
        'history': result['history'],
    }, output_dir / 'centralized_baseline.json')
    save_metrics_csv(result['history'], output_dir / 'centralized_history.csv')
    return result['model'], test_metrics


def evaluate_federated_test(model, test_loader, config, output_dir: Path):
    """Evaluate the final federated model once on the held-out test split."""
    device = config.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_metrics = evaluate_model(model, test_loader, device=device)
    test_metrics['test_loss'] = test_metrics.pop('validation_loss')
    test_metrics['test_accuracy'] = test_metrics.pop('validation_accuracy')
    test_metrics['dataset'] = config.dataset
    save_json(
        {'config': config.__dict__, 'test_metrics': test_metrics},
        output_dir / 'federated_test_metrics.json',
    )
    return test_metrics


def centralized_epoch_budget(config):
    """Match centralized sample exposure to the configured federated schedule."""
    selected_clients = max(1, int(round(config.client_fraction * config.num_clients)))
    participation_fraction = selected_clients / config.num_clients
    return max(1, int(round(config.rounds * config.local_epochs * participation_fraction)))


def ensure_unique_output_dir(base_dir: str) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        folder = base / f'experiment_{index:03d}'
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            return folder
        index += 1


def main():
    experiment_start = time.perf_counter()
    config = parse_config()
    set_seed(config.seed)
    output_dir = ensure_unique_output_dir(config.output_dir)

    if config.dataset == 'pathmnist':
        data_bundle = load_pathmnist_data(data_dir='data', download=True, batch_size=config.batch_size)
        if config.smoke_test:
            subset_indices = list(range(min(64, len(data_bundle['train_dataset']))))
            subset_factory = PathMNISTSubsetFactory(
                data_bundle['train_dataset'],
                batch_size=config.batch_size,
                sample_indices=subset_indices,
            )
            model = initialize_global_model('pathmnist', model_name=config.model, num_classes=9)
            fed_model, history = run_federated_training(subset_factory, data_bundle['val_loader'], config, output_dir=output_dir, global_model=model)
        else:
            dataset_factory = PathMNISTSubsetFactory(data_bundle['train_dataset'], batch_size=config.batch_size)
            model = initialize_global_model('pathmnist', model_name=config.model, num_classes=9)
            fed_model, history = run_federated_training(dataset_factory, data_bundle['val_loader'], config, output_dir=output_dir, global_model=model)

        save_json({'config': config.__dict__, 'history': history}, output_dir / 'federated_history.json')
        save_metrics_csv(history, output_dir / 'federated_history.csv')
        plot_round_metric(history, 'validation_accuracy', output_dir / 'global_validation_accuracy.png', 'Global validation accuracy vs. round')
        plot_round_metric(history, 'validation_loss', output_dir / 'global_validation_loss.png', 'Global validation loss vs. round')
        plot_round_metric(history, 'macro_f1', output_dir / 'macro_f1_vs_round.png', 'Macro F1-score vs. round')
        plot_round_metric(history, 'round_time_seconds', output_dir / 'training_time_vs_round.png', 'Training time vs. round')

        federated_test_metrics = evaluate_federated_test(
            fed_model,
            data_bundle['test_loader'],
            config,
            output_dir,
        )
        centralized_model, central_metrics = run_centralized_baseline(config, data_bundle, output_dir)
        save_json({
            'centralized_test': central_metrics,
            'federated_validation': history[-1] if history else {},
            'federated_test': federated_test_metrics,
        }, output_dir / 'comparison_summary.json')

    else:
        raise NotImplementedError('The IDS dataset integration is ready as an interface, but no concrete dataset is provided in this workspace yet.')

    save_json({'config': config.__dict__,
               'total_execution_time_seconds': time.perf_counter() - experiment_start,
               'includes': 'data loading, federated training, plots, held-out tests and centralized baseline'},
              output_dir / 'execution_summary.json')
    print(f'Experiment output directory: {output_dir}')


if __name__ == '__main__':
    main()
