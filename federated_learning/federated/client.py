from __future__ import annotations

import copy
import time
from typing import Iterable

import torch
from torch import nn


class FederatedClient:
    """Simulated federated client optimized for local training on a single machine."""

    def __init__(self, client_id: int, train_loader, model: nn.Module, device: str = 'cpu', optimizer_name: str = 'sgd', momentum: float = 0.9):
        self.client_id = client_id
        self.train_loader = train_loader
        self.model = model.to(device)
        self.device = device
        self.optimizer_name = optimizer_name.lower()
        self.momentum = momentum
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def load_global_model(self, state_dict):
        self.model.load_state_dict(copy.deepcopy(state_dict))

    def local_train(self, epochs: int, lr: float, optimizer_name: str | None = None):
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
        losses = []
        start_time = time.time()

        for _ in range(epochs):
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).view(-1).long()
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

        return {
            'client_id': self.client_id,
            'losses': losses,
            'train_time_seconds': time.time() - start_time,
            'state_dict': copy.deepcopy(self.model.state_dict()),
            'num_samples': len(self.train_loader.dataset),
        }


def local_train(model: nn.Module, train_loader, epochs: int, lr: float, device: str = 'cpu', optimizer_name: str = 'sgd', momentum: float = 0.9):
    """Train a single client from the current global model and return updated state."""
    client = FederatedClient(
        client_id=0,
        train_loader=train_loader,
        model=model,
        device=device,
        optimizer_name=optimizer_name,
        momentum=momentum,
    )
    return client.local_train(epochs=epochs, lr=lr, optimizer_name=optimizer_name)
