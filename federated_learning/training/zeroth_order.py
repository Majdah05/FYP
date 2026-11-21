"""Two-sided SPSA gradient estimates for ordinary PyTorch optimizers."""
from __future__ import annotations

import math

import torch
from torch import nn


class ZerothOrderEstimator:
    """Estimate gradients from Rademacher perturbations without autograd.

    Each minibatch uses two loss evaluations per direction and one unperturbed
    evaluation for logging and normal buffer updates. Perturbed evaluations use
    matching dropout randomness and leave model buffers unchanged.
    """

    def __init__(self, model: nn.Module, loss_fn, mu: float = 1e-3,
                 num_directions: int = 1, seed: int = 42):
        if isinstance(mu, bool) or not isinstance(mu, (int, float)) or not math.isfinite(mu) or mu <= 0:
            raise ValueError('zo_mu must be finite and positive.')
        if isinstance(num_directions, bool) or not isinstance(num_directions, int) or num_directions <= 0:
            raise ValueError('zo_directions must be a positive integer.')
        self.model = model
        self.loss_fn = loss_fn
        self.mu = mu
        self.num_directions = num_directions
        self.parameters = tuple(p for p in model.parameters() if p.requires_grad)
        if not self.parameters:
            raise ValueError('ZO requires at least one trainable parameter.')
        device = self.parameters[0].device
        if any(p.device != device or not p.is_floating_point() for p in self.parameters):
            raise ValueError('ZO requires real floating-point parameters on one device.')
        # Perturbations do not consume the global RNG used by dropout/data loading.
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.cuda_devices = [device.index] if device.type == 'cuda' else []

    @torch.no_grad()
    def estimate(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Populate parameter.grad and return the unperturbed minibatch loss."""
        originals = [p.detach().clone() for p in self.parameters]
        buffers = tuple(self.model.buffers())
        original_buffers = [buffer.detach().clone() for buffer in buffers]
        gradients = [torch.zeros_like(p) for p in self.parameters]

        try:
            for _ in range(self.num_directions):
                directions = [
                    torch.randint(0, 2, p.shape, device=p.device, generator=self.generator)
                    .to(p.dtype).mul_(2).sub_(1)
                    for p in self.parameters
                ]
                losses = []
                for sign in (1, -1):
                    for parameter, original, direction in zip(self.parameters, originals, directions):
                        parameter.copy_(original + sign * self.mu * direction)
                    for buffer, original in zip(buffers, original_buffers):
                        buffer.copy_(original)
                    # Both signs (and directions) see the same random masks.
                    # Global RNG advances only in the unperturbed forward below.
                    with torch.random.fork_rng(devices=self.cuda_devices):
                        loss = float(self.loss_fn(self.model(inputs), targets).item())
                    if not math.isfinite(loss):
                        raise FloatingPointError('Nonfinite ZO loss; check zo_mu and the learning rate.')
                    losses.append(loss)
                slope = (losses[0] - losses[1]) / (2 * self.mu)
                for gradient, direction in zip(gradients, directions):
                    gradient.add_(direction, alpha=slope / self.num_directions)
        finally:
            # Exact copies avoid accumulating roundoff from adding/subtracting mu.
            # Restoration also runs when a perturbed forward raises an exception.
            for parameter, original in zip(self.parameters, originals):
                parameter.copy_(original)
            for buffer, original in zip(buffers, original_buffers):
                buffer.copy_(original)

        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError('Nonfinite ZO gradient; check zo_mu and the learning rate.')
        loss = float(self.loss_fn(self.model(inputs), targets).item())
        if not math.isfinite(loss):
            raise FloatingPointError('Nonfinite unperturbed ZO loss.')
        for parameter, gradient in zip(self.parameters, gradients):
            parameter.grad = gradient
        return loss
