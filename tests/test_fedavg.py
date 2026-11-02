import copy
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from federated_learning.federated.aggregation import fedavg_aggregate
from federated_learning.federated.client import FederatedClient
from federated_learning.federated.server import FederatedServer, run_federated_training
from federated_learning.datasets.partition import partition_dataset
from federated_learning.training.evaluation import compute_metrics
from main import PathMNISTSubsetFactory, centralized_epoch_budget, evaluate_federated_test


def test_fedavg_aggregates_by_sample_weight():
    state_1 = {'layer.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 'layer.bias': torch.tensor([0.5, 1.5])}
    state_2 = {'layer.weight': torch.tensor([[5.0, 6.0], [7.0, 8.0]]), 'layer.bias': torch.tensor([2.0, 3.0])}

    result = fedavg_aggregate([state_1, state_2], [10, 30])

    expected_weight = (10 * state_1['layer.weight'] + 30 * state_2['layer.weight']) / 40
    expected_bias = (10 * state_1['layer.bias'] + 30 * state_2['layer.bias']) / 40

    assert torch.allclose(result['layer.weight'], expected_weight)
    assert torch.allclose(result['layer.bias'], expected_bias)


def test_client_starts_from_latest_global_model():
    model = torch.nn.Linear(2, 2)
    client = FederatedClient(client_id=0, train_loader=None, model=model, device='cpu')

    client.load_global_model(copy.deepcopy(model.state_dict()))
    assert client.model is not None
    assert client.model.state_dict()['weight'].shape == model.state_dict()['weight'].shape


def test_partial_client_participation_matches_fraction():
    server = FederatedServer(model=None, num_clients=10, client_fraction=0.4, seed=123)
    selected = server.select_clients()

    assert 0 < len(selected) <= 10
    assert len(set(selected)) == len(selected)
    assert len(selected) == 4


def test_evaluation_does_not_update_model_parameters():
    model = torch.nn.Linear(2, 2)
    original = copy.deepcopy(model.state_dict())
    x = torch.randn(4, 2)
    y = torch.randint(0, 2, (4,))
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    model.zero_grad()
    model.eval()

    client = FederatedClient(client_id=0, train_loader=None, model=model, device='cpu')
    client.model.load_state_dict(original)
    assert torch.allclose(client.model.state_dict()['weight'], original['weight'])


def test_binary_classification_auroc_uses_positive_class_probability():
    result = compute_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        np.array([[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]]),
    )

    assert result['auroc'] == 1.0


def test_smoke_factory_partitions_only_selected_samples():
    inputs = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    dataset = TensorDataset(inputs, targets)
    dataset.labels = targets.numpy()
    factory = PathMNISTSubsetFactory(dataset, batch_size=2, sample_indices=[1, 3, 5])

    assert factory.get_labels().tolist() == [1, 0, 2]
    selected_inputs, selected_targets = next(iter(factory.create_subset([2, 0])))
    assert selected_inputs.tolist() == [[10.0, 11.0], [2.0, 3.0]]
    assert selected_targets.tolist() == [2, 1]


def test_centralized_epoch_budget_matches_federated_participation():
    config = SimpleNamespace(rounds=20, local_epochs=2, num_clients=5, client_fraction=1.0)
    assert centralized_epoch_budget(config) == 40

    config.client_fraction = 0.4
    assert centralized_epoch_budget(config) == 16


def test_final_federated_test_evaluation_saves_metrics_without_updating_weights(tmp_path):
    model = torch.nn.Linear(2, 3)
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = torch.tensor([0, 1, 2])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=3)
    config = SimpleNamespace(dataset='pathmnist', device='cpu')
    state_before = copy.deepcopy(model.state_dict())

    results = evaluate_federated_test(model, loader, config, tmp_path)

    assert 'test_accuracy' in results
    assert 'test_loss' in results
    assert 'validation_accuracy' not in results
    assert (tmp_path / 'federated_test_metrics.json').exists()
    for key, value in state_before.items():
        assert torch.equal(model.state_dict()[key], value)


def test_server_applies_configured_local_epochs_and_optimizer(tmp_path):
    torch.manual_seed(13)
    inputs = torch.tensor([
        [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
        [-1.0, 1.0], [-1.0, 0.0], [0.5, -0.5],
    ])
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    dataset = TensorDataset(inputs, targets)

    class TinyFactory:
        def get_labels(self):
            return targets.numpy()

        def create_subset(self, indices):
            return DataLoader(Subset(dataset, indices), batch_size=2, shuffle=False)

    config = SimpleNamespace(
        device='cpu',
        output_dir=str(tmp_path),
        dataset='ids',
        model='mlp',
        num_clients=1,
        distribution='iid',
        dirichlet_alpha=0.5,
        seed=13,
        rounds=1,
        client_fraction=1.0,
        local_epochs=3,
        lr=0.05,
        momentum=0.9,
        optimizer='adam',
        save_checkpoints=False,
    )
    initial_model = torch.nn.Linear(2, 3)
    initial_state = copy.deepcopy(initial_model.state_dict())
    client_indices = partition_dataset(targets.numpy(), 1, 'iid', seed=config.seed)[0]
    expected_client = FederatedClient(
        client_id=0,
        train_loader=TinyFactory().create_subset(client_indices),
        model=copy.deepcopy(initial_model),
        device='cpu',
        optimizer_name='adam',
    )
    expected_client.load_global_model(initial_state)
    expected_result = expected_client.local_train(epochs=3, lr=config.lr, optimizer_name='adam')

    actual_model, history = run_federated_training(
        train_loader_factory=TinyFactory(),
        val_loader=DataLoader(dataset, batch_size=2),
        config=config,
        output_dir=tmp_path,
        global_model=copy.deepcopy(initial_model),
    )

    for key, expected_tensor in expected_result['state_dict'].items():
        assert torch.allclose(actual_model.state_dict()[key], expected_tensor)
    assert len(history) == 1
    assert history[0]['selected_client_ids'] == [0]
