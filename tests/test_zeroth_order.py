import copy
import builtins
import json
import tempfile

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

import main as experiment
from federated_learning.config import ExperimentConfig, parse_config
from federated_learning.federated.client import FederatedClient
from federated_learning.federated.server import run_federated_training
from federated_learning.models.cnn import PathMNISTCNN
from federated_learning.training.centralized import CentralizedTrainer
from federated_learning.training.zeroth_order import ZerothOrderEstimator


AGGREGATIONS = ['plaintext', pytest.param('ckks', marks=pytest.mark.he)]


def enable_he(aggregation, tmp_path, monkeypatch):
    if aggregation == 'ckks':
        pytest.importorskip('tenseal.sealapi')
        monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))


def reject_backward(*args, **kwargs):
    raise AssertionError('ZO training must not call backward().')


def tiny_dataset():
    inputs = torch.tensor([
        [1., 0.], [0., 1.], [-1., -1.],
        [2., 0.], [0., 2.], [-2., -2.],
        [0.5, 0.], [0., 0.5], [-0.5, -0.5],
        [1.5, 0.], [0., 1.5], [-1.5, -1.5],
    ])
    targets = torch.arange(len(inputs)) % 3
    return TensorDataset(inputs, targets)


class TinyFactory:
    def __init__(self, data):
        self.data = data

    def get_labels(self):
        return self.data.tensors[1].numpy()

    def create_subset(self, indices):
        return DataLoader(Subset(self.data, indices), batch_size=3)


@pytest.mark.parametrize('gradient_method', ['backprop', 'zo'])
@pytest.mark.parametrize('aggregation', ['plaintext', 'ckks'])
def test_configuration_supports_all_four_modes(gradient_method, aggregation):
    config = parse_config([
        '--gradient_method', gradient_method, '--optimizer', 'sgd',
        '--aggregation', aggregation, '--zo_mu', '0.002', '--zo_directions', '4',
    ])
    assert config.gradient_method == gradient_method
    assert config.aggregation == aggregation
    assert config.optimizer == 'sgd'
    assert config.zo_mu == 0.002 and config.zo_directions == 4
    assert parse_config([]).gradient_method == 'backprop'


def test_zo_yaml_and_explicit_cli_overrides(tmp_path):
    config_path = tmp_path / 'zo.yaml'
    config_path.write_text(
        'gradient_method: zo\noptimizer: sgd\naggregation: plaintext\n'
        'zo_mu: 0.002\nzo_directions: 3\n', encoding='utf-8',
    )
    config = parse_config(['--config', str(config_path), '--zo_directions', '7', '--aggregation', 'ckks'])
    assert config.gradient_method == 'zo' and config.zo_mu == 0.002
    assert config.zo_directions == 7 and config.aggregation == 'ckks'
    assert parse_config(['--config', str(config_path), '--gradient_method', 'backprop']).gradient_method == 'backprop'


@pytest.mark.parametrize('field,value', [
    ('gradient_method', 'unknown'),
    ('zo_mu', 0), ('zo_mu', -1), ('zo_mu', float('nan')), ('zo_mu', float('inf')),
    ('zo_mu', True), ('zo_mu', '0.001'),
    ('zo_directions', 0), ('zo_directions', -1), ('zo_directions', 1.5), ('zo_directions', True),
])
def test_invalid_zo_configuration(field, value):
    with pytest.raises(ValueError, match=field):
        ExperimentConfig(**{field: value}).validate()


@pytest.mark.parametrize('num_directions', [1, 4])
def test_two_sided_estimate_matches_scalar_quadratic_gradient(num_directions, monkeypatch):
    model = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(1.5)
    original = model.weight.detach().clone()
    monkeypatch.setattr(torch.Tensor, 'backward', reject_backward)
    estimator = ZerothOrderEstimator(model, nn.MSELoss(), num_directions=num_directions)
    loss = estimator.estimate(torch.tensor([[2.]], dtype=torch.float64), torch.zeros(1, 1).double())
    assert loss == pytest.approx(9.)
    assert torch.equal(model.weight, original)
    assert model.weight.grad.item() == pytest.approx(12., abs=1e-9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    assert model.weight.item() == pytest.approx(1.38, abs=1e-9)


def test_zo_matches_dropout_randomness_and_updates_buffers_only_once():
    class StochasticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.))
            self.register_buffer('forward_count', torch.tensor(0))
            self.masks = []

        def forward(self, inputs):
            self.forward_count.add_(1)
            mask = torch.rand_like(inputs) > 0.5
            self.masks.append(mask.clone())
            return self.weight * inputs * mask

    model = StochasticModel()
    inputs = torch.ones(4, 8)
    torch.manual_seed(31)
    initial_rng = torch.get_rng_state()
    torch.rand_like(inputs)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(initial_rng)
    estimator = ZerothOrderEstimator(model, nn.MSELoss(), num_directions=3, seed=9)
    estimator.estimate(inputs, torch.zeros_like(inputs))
    assert len(model.masks) == 7
    assert all(torch.equal(mask, model.masks[0]) for mask in model.masks)
    assert model.forward_count.item() == 1
    assert model.weight.item() == 2.
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_failed_probe_restores_parameters_buffers_and_rng():
    class FailingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.))
            self.register_buffer('counter', torch.tensor(0))
            self.calls = 0

        def forward(self, inputs):
            self.calls += 1
            self.counter.add_(1)
            outputs = self.weight * inputs * torch.rand_like(inputs)
            if self.calls == 2:
                raise RuntimeError('failed probe')
            return outputs

    model = FailingModel()
    state_before = copy.deepcopy(model.state_dict())
    rng_before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match='failed probe'):
        ZerothOrderEstimator(model, nn.MSELoss()).estimate(torch.ones(2), torch.zeros(2))
    assert all(torch.equal(model.state_dict()[name], value) for name, value in state_before.items())
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model.weight.grad is None


def test_zo_leaves_frozen_parameters_untouched():
    model = nn.Linear(2, 3)
    model.bias.requires_grad_(False)
    bias_before = model.bias.detach().clone()
    inputs, targets = tiny_dataset().tensors
    estimator = ZerothOrderEstimator(model, nn.CrossEntropyLoss(), num_directions=2)
    estimator.estimate(inputs, targets)
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert torch.equal(model.bias, bias_before)
    assert model.bias.grad is None
    assert model.weight.grad is not None


@pytest.mark.parametrize('optimizer_name', ['sgd', 'adam'])
def test_zo_client_and_centralized_training_reduce_loss_without_backward(optimizer_name, monkeypatch):
    torch.manual_seed(7)
    initial_model = nn.Linear(2, 3)
    data = tiny_dataset()
    loader = DataLoader(data, batch_size=len(data))
    inputs, targets = data.tensors
    initial_loss = nn.CrossEntropyLoss()(initial_model(inputs), targets).item()
    monkeypatch.setattr(torch.Tensor, 'backward', reject_backward)
    client = FederatedClient(
        0, loader, copy.deepcopy(initial_model), optimizer_name=optimizer_name,
        momentum=0., gradient_method='zo', zo_directions=16, zo_seed=5,
    )
    result = client.local_train(epochs=4, lr=0.05)
    assert len(result['losses']) == 4
    assert nn.CrossEntropyLoss()(client.model(inputs), targets).item() < initial_loss
    trainer = CentralizedTrainer(
        copy.deepcopy(initial_model), optimizer_name=optimizer_name, lr=0.05,
        momentum=0., gradient_method='zo', zo_directions=16, zo_seed=5,
    )
    centralized = trainer.train(loader, loader, epochs=4)
    assert centralized['history'][-1]['validation_loss'] < initial_loss
    assert all(row['gradient_method'] == 'zo' for row in centralized['history'])
    for name, value in client.model.state_dict().items():
        assert torch.equal(value, trainer.model.state_dict()[name])


def test_zo_client_training_is_reproducible():
    torch.manual_seed(19)
    initial = nn.Sequential(nn.Linear(2, 8), nn.Dropout(0.2), nn.Linear(8, 3))
    data = tiny_dataset()
    states = []
    for _ in range(2):
        torch.manual_seed(29)
        client = FederatedClient(
            2, DataLoader(data, batch_size=4), copy.deepcopy(initial),
            gradient_method='zo', zo_directions=3, zo_seed=123,
        )
        states.append(client.local_train(epochs=2, lr=0.001)['state_dict'])
    assert all(torch.equal(states[0][name], states[1][name]) for name in states[0])


def test_pathmnist_cnn_zo_step_without_backward(monkeypatch):
    torch.manual_seed(8)
    model = PathMNISTCNN()
    original = model.classifier[-1].weight.detach().clone()
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, 28, 28), torch.tensor([0, 1])), batch_size=2,
    )
    monkeypatch.setattr(torch.Tensor, 'backward', reject_backward)
    client = FederatedClient(0, loader, model, gradient_method='zo', zo_directions=2)
    result = client.local_train(epochs=1, lr=0.0001)
    assert len(result['losses']) == 1
    assert all(torch.isfinite(value).all() for value in result['state_dict'].values())
    assert not torch.equal(model.classifier[-1].weight, original)


def test_plaintext_zo_runs_without_importing_optional_he(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == 'tenseal' or name.startswith('tenseal.'):
            raise AssertionError('Plaintext ZO must not import TenSEAL.')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(torch.Tensor, 'backward', reject_backward)
    config = parse_config([
        '--gradient_method', 'zo', '--aggregation', 'plaintext',
        '--num_clients', '3', '--rounds', '1', '--local_epochs', '1', '--device', 'cpu',
    ])
    data = tiny_dataset()
    _, history = run_federated_training(
        TinyFactory(data), DataLoader(data, batch_size=4), config, tmp_path, nn.Linear(2, 3),
    )
    assert history[0]['aggregation'] == 'plaintext'


@pytest.mark.parametrize('gradient_method', ['backprop', 'zo'])
@pytest.mark.parametrize('aggregation', AGGREGATIONS)
@pytest.mark.parametrize('distribution', ['iid', 'non_iid'])
def test_federated_modes_with_partial_participation(
        gradient_method, aggregation, distribution, tmp_path, monkeypatch):
    enable_he(aggregation, tmp_path, monkeypatch)
    torch.manual_seed(17)
    data = tiny_dataset()
    initial = nn.Linear(2, 3)
    state_before = copy.deepcopy(initial.state_dict())
    if gradient_method == 'zo':
        monkeypatch.setattr(torch.Tensor, 'backward', reject_backward)
    config = ExperimentConfig(
        num_clients=3, rounds=2, local_epochs=1, batch_size=3, lr=0.001,
        client_fraction=2/3, device='cpu', aggregation=aggregation,
        distribution=distribution, gradient_method=gradient_method, zo_directions=2,
    )
    model, history = run_federated_training(
        TinyFactory(data), DataLoader(data, batch_size=4), config, tmp_path, initial,
    )
    assert len(history) == 2
    assert all(len(row['selected_client_ids']) == 2 for row in history)
    assert all(row['gradient_method'] == gradient_method for row in history)
    assert all(row['aggregation'] == aggregation for row in history)
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    assert any(not torch.equal(value, state_before[name]) for name, value in model.state_dict().items())
    checkpoint = torch.load(tmp_path / 'checkpoint_round_2.pt', weights_only=False)
    assert checkpoint['config']['gradient_method'] == gradient_method
    assert checkpoint['config']['zo_directions'] == 2
    if aggregation == 'ckks':
        assert all(row['he_ciphertext_upload_bytes'] > 0 for row in history)


@pytest.mark.he
def test_zo_ckks_matches_plaintext_training_and_hides_client_state(tmp_path, monkeypatch):
    enable_he('ckks', tmp_path, monkeypatch)
    monkeypatch.setattr(torch.Tensor, 'backward', reject_backward)
    original_local_train = FederatedClient.local_train
    encrypted_uploads = []

    def inspect_upload(self, *args, **kwargs):
        result = original_local_train(self, *args, **kwargs)
        if kwargs.get('update_encoder') is not None:
            assert 'state_dict' not in result
            assert all(isinstance(chunk, bytes) for chunk in result['encrypted_update'].chunks)
            encrypted_uploads.append(result['encrypted_update'])
        return result

    monkeypatch.setattr(FederatedClient, 'local_train', inspect_upload)
    torch.manual_seed(4)
    initial = nn.Linear(2, 3)
    data = tiny_dataset()
    results = []
    for aggregation in ('plaintext', 'ckks'):
        config = ExperimentConfig(
            num_clients=3, rounds=2, local_epochs=1, lr=0.001, device='cpu',
            gradient_method='zo', zo_directions=3, aggregation=aggregation,
        )
        torch.manual_seed(33)
        model, _ = run_federated_training(
            TinyFactory(data), DataLoader(data, batch_size=4), config,
            tmp_path / aggregation, copy.deepcopy(initial),
        )
        results.append(model.state_dict())
    assert len(encrypted_uploads) == 6
    for name in results[0]:
        assert torch.allclose(results[0][name], results[1][name], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('aggregation', AGGREGATIONS)
def test_main_zo_run_includes_centralized_baseline(aggregation, tmp_path, monkeypatch):
    enable_he(aggregation, tmp_path, monkeypatch)
    data = tiny_dataset()
    data.labels = data.tensors[1].numpy()
    loader = DataLoader(data, batch_size=4)
    bundle = {
        'train_dataset': data, 'train_loader': loader,
        'val_loader': loader, 'test_loader': loader,
    }
    monkeypatch.setattr(experiment, 'load_pathmnist_data', lambda **kwargs: bundle)
    monkeypatch.setattr(experiment, 'initialize_global_model', lambda *args, **kwargs: nn.Linear(2, 3))
    monkeypatch.setattr(experiment, 'plot_round_metric', lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.Tensor, 'backward', reject_backward)
    monkeypatch.setattr('sys.argv', [
        'main.py', '--gradient_method', 'zo', '--optimizer', 'sgd',
        '--aggregation', aggregation, '--num_clients', '3', '--rounds', '1',
        '--local_epochs', '1', '--batch_size', '4', '--lr', '0.001',
        '--zo_directions', '2', '--device', 'cpu', '--smoke_test',
        '--output_dir', str(tmp_path / 'run'),
    ])
    experiment.main()
    output = tmp_path / 'run' / 'experiment_001'
    centralized = json.loads((output / 'centralized_baseline.json').read_text())
    assert centralized['config']['gradient_method'] == 'zo'
    assert centralized['history'][0]['gradient_method'] == 'zo'
    assert (output / 'federated_test_metrics.json').exists()
    assert (output / 'comparison_summary.json').exists()
    saved = json.loads((output / 'experiment_config.json').read_text())
    assert saved['aggregation']['aggregation'] == aggregation
    assert saved['config']['zo_directions'] == 2
