from __future__ import annotations

import numpy as np


def _validate_num_clients(num_clients: int, total_samples: int) -> int:
    if num_clients <= 0:
        raise ValueError('num_clients must be positive.')
    if num_clients > total_samples:
        raise ValueError(
            f'num_clients={num_clients} cannot exceed the number of samples={total_samples}. '
            'Reduce the client count or use a larger dataset.'
        )
    return num_clients


def partition_dataset(labels, num_clients: int, distribution: str = 'iid', dirichlet_alpha: float = 0.5, seed: int = 42):
    """Partition a label array across clients.

    Parameters
    ----------
    labels : np.ndarray or sequence
        Integer class labels for all training samples.
    num_clients : int
        Number of simulated clients.
    distribution : {'iid', 'non_iid'}
        Partitioning strategy.
    dirichlet_alpha : float
        Concentration for Dirichlet-based label skew.
    seed : int
        Random seed for reproducibility.
    """
    labels = np.asarray(labels)
    total_samples = len(labels)
    num_clients = _validate_num_clients(num_clients, total_samples)

    if distribution == 'iid':
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(total_samples)
        chunks = np.array_split(shuffled, num_clients)
        return [chunk.tolist() for chunk in chunks if len(chunk) > 0]

    if distribution == 'non_iid':
        if dirichlet_alpha <= 0:
            raise ValueError('dirichlet_alpha must be positive for non-IID partitioning.')
        classes = np.unique(labels)
        rng = np.random.default_rng(seed)
        client_indices = [[] for _ in range(num_clients)]

        for label in classes:
            class_indices = np.where(labels == label)[0]
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, dirichlet_alpha), size=1)[0]
            client_counts = rng.multinomial(len(class_indices), proportions)
            start = 0
            for client_id, count in enumerate(client_counts):
                end = start + int(count)
                client_indices[client_id].extend(class_indices[start:end].tolist())
                start = end

        # Repair empty clients by moving, not copying, a sample from a larger client.
        for client_id in range(num_clients):
            if not client_indices[client_id]:
                donor_id = max(range(num_clients), key=lambda index: len(client_indices[index]))
                donor_position = int(rng.integers(len(client_indices[donor_id])))
                client_indices[client_id].append(client_indices[donor_id].pop(donor_position))

        return [sorted(client) for client in client_indices]

    raise ValueError(f"Unknown distribution '{distribution}'. Choose 'iid' or 'non_iid'.")
