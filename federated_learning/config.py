from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    dataset: str = 'pathmnist'
    num_clients: int = 5
    rounds: int = 20
    local_epochs: int = 2
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    optimizer: str = 'sgd'
    distribution: str = 'iid'
    dirichlet_alpha: float = 0.5
    client_fraction: float = 1.0
    model: str = 'cnn'
    seed: int = 42
    device: str = 'auto'
    output_dir: str = 'results'
    save_checkpoints: bool = True
    smoke_test: bool = False
    dataset_path: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        valid_datasets = {'pathmnist', 'ids'}
        if self.dataset not in valid_datasets:
            raise ValueError(f"Unsupported dataset '{self.dataset}'. Allowed: {sorted(valid_datasets)}")
        if self.num_clients <= 0:
            raise ValueError('num_clients must be positive.')
        if self.rounds <= 0:
            raise ValueError('rounds must be positive.')
        if self.local_epochs <= 0:
            raise ValueError('local_epochs must be positive.')
        if self.batch_size <= 0:
            raise ValueError('batch_size must be positive.')
        if self.lr <= 0:
            raise ValueError('lr must be positive.')
        if not (0 <= self.momentum < 1):
            raise ValueError('momentum must be in [0,1).')
        if self.optimizer.lower() not in {'sgd', 'adam'}:
            raise ValueError("optimizer must be 'sgd' or 'adam'.")
        if self.distribution not in {'iid', 'non_iid'}:
            raise ValueError("distribution must be 'iid' or 'non_iid'.")
        if not (0 < self.client_fraction <= 1):
            raise ValueError('client_fraction must be in (0,1].')
        if self.dirichlet_alpha <= 0:
            raise ValueError('dirichlet_alpha must be positive.')
        if self.model not in {'cnn', 'mlp'}:
            raise ValueError("model must be 'cnn' or 'mlp'.")
        if self.device not in {'cpu', 'cuda', 'auto'}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'.")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f'YAML config not found: {config_path}')
    with config_path.open('r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Configurable federated learning baseline using FedAvg.')
    parser.add_argument('--dataset', type=str, default='pathmnist', choices=['pathmnist', 'ids'])
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--local_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--distribution', type=str, default='iid', choices=['iid', 'non_iid'])
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5)
    parser.add_argument('--client_fraction', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mlp'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--save_checkpoints', action='store_true')
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--config', type=str, default=None, help='Optional YAML config file to load before CLI overrides.')
    parser.add_argument('--dataset_path', type=str, default=None, help='Optional dataset path for custom IDS datasets.')
    return parser


def merge_config(cli_args: argparse.Namespace, yaml_data: dict[str, Any] | None = None) -> ExperimentConfig:
    base = yaml_data or {}
    merged = {**base, **vars(cli_args)}
    if 'config' in merged:
        merged.pop('config', None)
    cfg = ExperimentConfig(**merged)
    cfg.validate()
    return cfg


def parse_config(argv: list[str] | None = None) -> ExperimentConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    yaml_data = load_yaml_config(args.config) if args.config else {}
    return merge_config(args, yaml_data)
