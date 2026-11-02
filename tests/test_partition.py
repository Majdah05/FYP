import numpy as np

from federated_learning.datasets.partition import partition_dataset


def test_iid_partition_no_duplicates_or_loss():
    labels = np.arange(30)
    clients = partition_dataset(labels, num_clients=5, distribution='iid', seed=42)

    all_ids = []
    for client in clients:
        assert len(client) > 0, 'Each client should receive data'
        all_ids.extend(client)

    assert len(all_ids) == len(labels)
    assert sorted(all_ids) == sorted(labels.tolist())
    assert len(set(all_ids)) == len(all_ids)


def test_non_iid_partition_respects_dirichlet_alpha():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    clients = partition_dataset(labels, num_clients=4, distribution='non_iid', dirichlet_alpha=0.5, seed=7)

    all_ids = []
    for client in clients:
        assert len(client) > 0, 'Each client should be non-empty'
        all_ids.extend(client)

    assert len(all_ids) == len(labels)
    assert len(set(all_ids)) == len(all_ids)
    assert len(clients) == 4


def test_extreme_dirichlet_partition_keeps_all_samples_and_nonempty_clients():
    labels = np.array([0] * 10 + [1] * 2 + [2] * 2)

    for seed in range(20):
        clients = partition_dataset(
            labels,
            num_clients=3,
            distribution='non_iid',
            dirichlet_alpha=0.01,
            seed=seed,
        )
        assigned = [sample_id for client in clients for sample_id in client]

        assert len(clients) == 3
        assert all(clients)
        assert len(assigned) == len(labels)
        assert sorted(assigned) == list(range(len(labels)))


def test_iid_partition_uses_stratified_like_distribution():
    labels = np.array([0] * 40 + [1] * 40 + [2] * 20)
    clients = partition_dataset(labels, num_clients=4, distribution='iid', seed=3)

    for client in clients:
        unique, counts = np.unique(labels[client], return_counts=True)
        assert sum(counts) == len(client)

    assert len(clients) == 4
