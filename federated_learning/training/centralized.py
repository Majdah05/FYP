from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from torch import nn

from federated_learning.training.evaluation import evaluate_model
from federated_learning.training.zeroth_order import ZerothOrderEstimator


class CentralizedTrainer:
    """Basic centralized training loop for comparison against FL."""

    def __init__(self, model: nn.Module, device: str = 'cpu', optimizer_name: str = 'sgd',
                 lr: float = 0.01, momentum: float = 0.9,
                 gradient_method: str = 'backprop', zo_mu: float = 1e-3,
                 zo_directions: int = 1, zo_seed: int = 42):
        if gradient_method not in {'backprop', 'zo'}:
            raise ValueError("gradient_method must be 'backprop' or 'zo'.")
        self.model = model
        self.device = device
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.momentum = momentum
        self.gradient_method = gradient_method
        self.zo_mu = zo_mu
        self.zo_directions = zo_directions
        self.zo_seed = zo_seed
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, train_loader, val_loader, epochs: int = 5):
        device = torch.device(self.device)
        self.model.to(device)
        optimizer_cls = torch.optim.Adam if self.optimizer_name == 'adam' else torch.optim.SGD
        if self.optimizer_name == 'adam':
            optimizer = optimizer_cls(self.model.parameters(), lr=self.lr)
        else:
            optimizer = optimizer_cls(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        zo_estimator = None
        if self.gradient_method == 'zo':
            zo_estimator = ZerothOrderEstimator(
                self.model, self.criterion, self.zo_mu, self.zo_directions, self.zo_seed,
            )

        history = []
        start_time = time.time()
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).view(-1).long()
                optimizer.zero_grad()
                if zo_estimator is None:
                    logits = self.model(inputs)
                    loss = self.criterion(logits, targets)
                    loss.backward()
                    loss_value = float(loss.item())
                else:
                    loss_value = zo_estimator.estimate(inputs, targets)
                optimizer.step()
                epoch_loss += loss_value

            val_metrics = evaluate_model(self.model, val_loader, device=str(device))
            val_metrics['epoch'] = epoch + 1
            val_metrics['training_loss'] = epoch_loss / max(1, len(train_loader))
            val_metrics['gradient_method'] = self.gradient_method
            val_metrics['optimizer'] = self.optimizer_name
            val_metrics['epoch_time_seconds'] = time.time() - start_time
            history.append(val_metrics)

        result = {
            'model': self.model,
            'history': history,
            'total_time_seconds': time.time() - start_time,
        }
        return result
