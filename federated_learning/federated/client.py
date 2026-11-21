from __future__ import annotations

import copy
import time
from typing import Iterable

import torch
from torch import nn

from federated_learning.training.zeroth_order import ZerothOrderEstimator


class FederatedClient:
    """Simulated federated client optimized for local training on a single machine."""

    def __init__(self, client_id: int, train_loader, model: nn.Module, device: str = 'cpu',
                 optimizer_name: str = 'sgd', momentum: float = 0.9,
                 gradient_method: str = 'backprop', zo_mu: float = 1e-3,
                 zo_directions: int = 1, zo_seed: int = 42):
        if gradient_method not in {'backprop', 'zo'}:
            raise ValueError("gradient_method must be 'backprop' or 'zo'.")
        self.client_id = client_id
        self.train_loader = train_loader
        self.model = model.to(device)
        self.device = device
        self.optimizer_name = optimizer_name.lower()
        self.momentum = momentum
        self.gradient_method = gradient_method
        self.zo_mu = zo_mu
        self.zo_directions = zo_directions
        self.zo_seed = zo_seed
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def load_global_model(self, state_dict):
        self.model.load_state_dict(copy.deepcopy(state_dict))

    def local_train(self, epochs: int, lr: float, optimizer_name: str | None = None,
                    update_encoder=None, round_id: int = 0):
        if self.train_loader is None:
            raise ValueError(f'Client {self.client_id} has no training data.')

        selected_optimizer = (optimizer_name or self.optimizer_name).lower()
        if selected_optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif selected_optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=self.momentum)
        else:
            raise ValueError(f"Unsupported optimizer '{selected_optimizer}'. Choose 'sgd' or 'adam'.")
        self.model.train()
        zo_estimator = None
        if self.gradient_method == 'zo':
            zo_estimator = ZerothOrderEstimator(
                self.model, self.loss_fn, self.zo_mu, self.zo_directions, self.zo_seed,
            )
        losses = []
        start_time = time.perf_counter()

        for _ in range(epochs):
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).view(-1).long()
                optimizer.zero_grad()
                if zo_estimator is None:
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    loss_value = float(loss.item())
                else:
                    loss_value = zo_estimator.estimate(inputs, targets)
                optimizer.step()
                losses.append(loss_value)

        result = {
            'client_id': self.client_id,
            'losses': losses,
            'train_time_seconds': time.perf_counter() - start_time,
            'state_dict': copy.deepcopy(self.model.state_dict()),
            'num_samples': len(self.train_loader.dataset),
        }
        if update_encoder is not None:
            # Packaging occurs inside the client role. The coordinator receives
            # ciphertext bytes and public metadata, never this plaintext state.
            state = result.pop('state_dict')
            result['encrypted_update'], result['he_metrics'] = update_encoder.prepare(
                state, result['num_samples'], self.client_id, round_id,
            )
        return result


def local_train(model: nn.Module, train_loader, epochs: int, lr: float, device: str = 'cpu',
                optimizer_name: str = 'sgd', momentum: float = 0.9,
                gradient_method: str = 'backprop', zo_mu: float = 1e-3,
                zo_directions: int = 1, zo_seed: int = 42):
    """Train a single client from the current global model and return updated state."""
    client = FederatedClient(
        client_id=0,
        train_loader=train_loader,
        model=model,
        device=device,
        optimizer_name=optimizer_name,
        momentum=momentum,
        gradient_method=gradient_method,
        zo_mu=zo_mu,
        zo_directions=zo_directions,
        zo_seed=zo_seed,
    )
    return client.local_train(epochs=epochs, lr=lr, optimizer_name=optimizer_name)
