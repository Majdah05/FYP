from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from federated_learning.datasets.partition import partition_dataset
from federated_learning.federated.strategies import create_aggregation_strategy
from federated_learning.federated.client import FederatedClient
from federated_learning.models.cnn import PathMNISTCNN
from federated_learning.models.mlp import TabularMLP
from federated_learning.training.evaluation import compute_metrics, evaluate_model as evaluate_model_metrics
from federated_learning.utils.logging import save_json


class FederatedServer:
    """Simple server orchestration for a single-machine federated simulation."""

    def __init__(self, model: nn.Module | None, num_clients: int, client_fraction: float = 1.0, seed: int = 42):
        self.model = model
        self.num_clients = num_clients
        self.client_fraction = client_fraction
        self.seed = seed

    def select_clients(self):
        return select_clients(self.num_clients, self.client_fraction, seed=self.seed)


def initialize_global_model(dataset: str, model_name: str | None = None, input_dim: int | None = None, num_classes: int = 9):
    """Create a fresh global model matching the dataset type."""
    if model_name is None:
        model_name = 'cnn' if dataset == 'pathmnist' else 'mlp'

    if model_name.lower() == 'cnn':
        return PathMNISTCNN(num_classes=num_classes)
    if model_name.lower() == 'mlp':
        if input_dim is None:
            raise ValueError('input_dim must be provided for MLP models.')
        return TabularMLP(input_dim=input_dim, num_classes=num_classes)
    raise ValueError(f'Unsupported model name: {model_name}')


def select_clients(num_clients: int, client_fraction: float, seed: int = 42):
    """Select a subset of clients based on client_fraction."""
    if not (0 < client_fraction <= 1):
        raise ValueError('client_fraction must be within (0, 1].')
    selected_count = max(1, int(round(client_fraction * num_clients)))
    if selected_count > num_clients:
        selected_count = num_clients
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.arange(num_clients), size=selected_count, replace=False).tolist())


def evaluate_model(model: nn.Module, data_loader, device: str = 'cpu', criterion=None):
    """Evaluate a model on a validation or test loader without updating the model."""
    return evaluate_model_metrics(model, data_loader, device=device, criterion=criterion)


def _resolve_device(device: str) -> str:
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def run_federated_training(
    train_loader_factory,
    val_loader,
    config,
    output_dir: str | None = None,
    global_model: nn.Module | None = None,
):
    """Run the FedAvg training loop for a configurable experiment."""
    device = _resolve_device(config.device)
    output_dir = Path(output_dir or config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if global_model is None:
        global_model = initialize_global_model(config.dataset, model_name=config.model, input_dim=64, num_classes=9)
    global_model = global_model.to(device)
    global_state = copy.deepcopy(global_model.state_dict())
    experiment_start = time.perf_counter()
    strategy = create_aggregation_strategy(config, global_state)
    save_json({'config': config.__dict__, 'aggregation': strategy.metadata}, output_dir / 'experiment_config.json')

    # Partition the training set only.
    labels = train_loader_factory.get_labels()
    partitions = partition_dataset(
        labels,
        num_clients=config.num_clients,
        distribution=config.distribution,
        dirichlet_alpha=config.dirichlet_alpha,
        seed=config.seed,
    )

    client_loaders = []
    for client_id, indices in enumerate(partitions):
        subset = train_loader_factory.create_subset(indices)
        client_loaders.append((client_id, subset))

    rounds_history = []
    start_time = experiment_start

    for round_idx in range(config.rounds):
        round_start = time.perf_counter()
        selected_clients = select_clients(config.num_clients, config.client_fraction, seed=config.seed + round_idx)
        updated_states = []
        sample_counts = []
        round_losses = []
        client_he_metrics = []
        strategy.begin_round(round_idx + 1, tuple(
            (client_id, len(client_loaders[client_id][1].dataset)) for client_id in selected_clients
        ))

        for client_id in selected_clients:
            client_loader = client_loaders[client_id][1]
            client_model = copy.deepcopy(global_model)
            client_model.to(device)
            client = FederatedClient(
                client_id=client_id,
                train_loader=client_loader,
                model=client_model,
                device=device,
                optimizer_name=config.optimizer,
                momentum=config.momentum,
            )
            client.load_global_model(global_state)
            local_result = client.local_train(
                epochs=config.local_epochs,
                lr=config.lr,
                optimizer_name=config.optimizer,
                update_encoder=strategy.client_encoder,
                round_id=round_idx + 1,
            )
            round_losses.append(float(np.mean(local_result['losses'])))
            if strategy.client_encoder is None:
                updated_states.append(local_result['state_dict'])
            else:
                updated_states.append(local_result['encrypted_update'])
                client_he_metrics.append(local_result['he_metrics'])
            sample_counts.append(local_result['num_samples'])
            del local_result, client, client_model

        global_state, he_metrics = strategy.aggregate(updated_states, sample_counts, client_he_metrics)
        global_model.load_state_dict(global_state)

        metrics = evaluate_model(global_model, val_loader, device=device)
        metrics['round'] = round_idx + 1
        metrics['participating_clients'] = selected_clients
        metrics['selected_client_ids'] = selected_clients
        metrics['training_loss'] = float(np.mean(round_losses))
        metrics['per_client_training_loss'] = round_losses
        metrics.update(he_metrics)
        metrics['aggregation'] = getattr(config, 'aggregation', 'plaintext')
        metrics['round_time_seconds'] = time.perf_counter() - round_start
        metrics['total_experiment_time_seconds'] = time.perf_counter() - start_time
        metrics['communication_volume_bytes'] = estimate_communication_cost(selected_clients, global_model)
        metrics['communication_volume_basis'] = 'legacy estimated plaintext model upload plus download; not measured network traffic'
        metrics['plaintext_float_upload_bytes_estimate'] = len(selected_clients) * sum(
            value.numel() * value.element_size() for value in global_state.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        )
        rounds_history.append(metrics)

        if config.save_checkpoints:
            checkpoint_path = output_dir / f'checkpoint_round_{round_idx + 1}.pt'
            torch.save({'round': round_idx + 1, 'model_state': global_model.state_dict(), 'config': config.__dict__}, checkpoint_path)

    save_json({'config': config.__dict__, 'aggregation': strategy.metadata,
               'federated_execution_time_seconds': time.perf_counter() - start_time}, output_dir / 'experiment_config.json')
    return global_model, rounds_history


def estimate_communication_cost(selected_clients, model):
    """Estimate the bytes transferred by a client-server exchange for one round."""
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_float = 4
    transfer_per_direction = total_params * bytes_per_float
    return int((len(selected_clients) * transfer_per_direction) * 2)


def compute_average_round_metrics(history):
    if not history:
        return {}
    keys = ['validation_loss', 'validation_accuracy', 'macro_f1']
    summary = {}
    for key in keys:
        values = [entry.get(key, 0.0) for entry in history]
        summary[key] = float(np.mean(values))
    summary['rounds'] = len(history)
    return summary
