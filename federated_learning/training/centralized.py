from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from torch import nn

from federated_learning.training.evaluation import evaluate_model


class CentralizedTrainer:
    """Basic centralized training loop for comparison against FL."""

    def __init__(self, model: nn.Module, device: str = 'cpu', optimizer_name: str = 'sgd', lr: float = 0.01, momentum: float = 0.9):
        self.model = model
        self.device = device
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.momentum = momentum
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, train_loader, val_loader, epochs: int = 5):
        device = torch.device(self.device)
        self.model.to(device)
        optimizer_cls = torch.optim.Adam if self.optimizer_name == 'adam' else torch.optim.SGD
        if self.optimizer_name == 'adam':
            optimizer = optimizer_cls(self.model.parameters(), lr=self.lr)
        else:
            optimizer = optimizer_cls(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        history = []
        start_time = time.time()
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).view(-1).long()
                optimizer.zero_grad()
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

            val_metrics = evaluate_model(self.model, val_loader, device=str(device))
            val_metrics['epoch'] = epoch + 1
            val_metrics['training_loss'] = epoch_loss / max(1, len(train_loader))
            val_metrics['epoch_time_seconds'] = time.time() - start_time
            history.append(val_metrics)

        result = {
            'model': self.model,
            'history': history,
            'total_time_seconds': time.time() - start_time,
        }
        return result
