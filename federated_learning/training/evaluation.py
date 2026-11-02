from __future__ import annotations

import numpy as np
import torch
from sklearn import metrics
from torch import nn


def compute_metrics(y_true, y_pred, y_prob=None, average='macro', labels=None):
    accuracy = float(np.mean(y_true == y_pred))
    f1 = float(metrics.f1_score(y_true, y_pred, average=average, labels=labels, zero_division=0))
    precision = float(metrics.precision_score(y_true, y_pred, average=average, labels=labels, zero_division=0))
    recall = float(metrics.recall_score(y_true, y_pred, average=average, labels=labels, zero_division=0))
    roc_auc = None
    if y_prob is not None and y_prob.ndim == 2 and len(np.unique(y_true)) >= 2:
        try:
            if y_prob.shape[1] == 2:
                roc_auc = float(metrics.roc_auc_score(y_true, y_prob[:, 1]))
            else:
                roc_auc = float(metrics.roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
        except ValueError:
            roc_auc = None
    elif y_prob is not None and y_prob.ndim == 1 and len(np.unique(y_true)) == 2:
        roc_auc = float(metrics.roc_auc_score(y_true, y_prob))
    return {
        'accuracy': accuracy,
        'macro_f1': f1,
        'precision': precision,
        'recall': recall,
        'auroc': roc_auc,
    }


def evaluate_model(model: nn.Module, data_loader, device: str = 'cpu', criterion=None):
    """Evaluate a model without altering parameters."""
    device = torch.device(device)
    model.eval()
    criterion = criterion or torch.nn.CrossEntropyLoss()

    outputs_all = []
    labels_all = []
    losses = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1).long()
            logits = model(inputs)
            loss = criterion(logits, targets)
            losses.append(float(loss.item()))
            outputs_all.append(logits.cpu())
            labels_all.append(targets.cpu())

    logits = torch.cat(outputs_all, dim=0) if outputs_all else torch.empty(0)
    labels = torch.cat(labels_all, dim=0) if labels_all else torch.empty(0, dtype=torch.long)
    preds = logits.argmax(dim=1)
    y_true = labels.numpy()
    y_pred = preds.numpy()
    y_prob = torch.softmax(logits, dim=1).numpy() if logits.numel() > 0 else np.empty((0, 0))
    metric_values = compute_metrics(y_true, y_pred, y_prob)
    metric_values['validation_loss'] = float(np.mean(losses)) if losses else 0.0
    metric_values['validation_accuracy'] = metric_values['accuracy']
    return metric_values
