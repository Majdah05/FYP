"""Real Microsoft SEAL tests. Run: python -m pytest -m he --basetemp .test-he"""
import copy
import dataclasses
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from federated_learning.config import ExperimentConfig
from federated_learning.encryption.ckks_context import PublicContext, create_context
from federated_learning.encryption.ckks_encoder import CKKSEncoder, load_ciphertext, state_schema
from federated_learning.encryption.interface import EncryptedUpdate
from federated_learning.federated.aggregation import fedavg_aggregate
from federated_learning.federated.client import FederatedClient
from federated_learning.federated.server import run_federated_training
from federated_learning.federated.strategies import create_aggregation_strategy

pytestmark = pytest.mark.he


@pytest.fixture(autouse=True)
def he_temp_directory(tmp_path, monkeypatch):
    pytest.importorskip('tenseal.sealapi')
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))


@pytest.fixture
def state():
    return {'weight': torch.tensor([[-1.5, 0.0, 1.e-6], [0.25, 2.125, -0.01]], dtype=torch.float64),
            'bias': torch.tensor([0.123, -0.2], dtype=torch.float32),
            'empty': torch.empty(0, 2), 'counter': torch.tensor(7, dtype=torch.int64)}


def strategy_for(state, **kwargs):
    return create_aggregation_strategy(ExperimentConfig(aggregation='ckks', **kwargs), state)


def aggregate_states(strategy, states, counts, round_id=1):
    strategy.begin_round(round_id, tuple(enumerate(counts)))
    packaged = [strategy.client_encoder.prepare(s, n, i, round_id) for i, (s, n) in enumerate(zip(states, counts))]
    return strategy.aggregate([p[0] for p in packaged], counts, [p[1] for p in packaged])


def test_context_public_serialization_and_no_evaluation_keys(state):
    strategy = strategy_for(state)
    public = PublicContext(strategy.client_encoder.public.bundle)
    assert public.context.parameters_set()
    assert public.slot_count == 4096
    assert not hasattr(public, 'secret_key')
    assert not hasattr(strategy.server, 'decryptor')
    assert strategy.metadata['relinearization_keys_generated'] is False
    assert strategy.metadata['galois_keys_generated'] is False
    # Public-key encryption cannot be used as a secret decryption key.
    with pytest.raises(TypeError):
        public.seal.Decryptor(public.context, public.public_key)
    with pytest.raises((ValueError, RuntimeError)):
        public.seal.Decryptor(public.context, public.seal.SecretKey())


def test_round_trip_and_addition_only_unequal_weighting(state, monkeypatch):
    strategy = strategy_for(state, ckks_chunk_size=4)
    states = [copy.deepcopy(state) for _ in range(3)]
    for i, s in enumerate(states):
        s['weight'] += i * 0.333
        s['bias'] -= i * 0.25
        s['counter'] += i
    counts = [20, 10, 30]
    calls = []
    additions = []
    native_evaluator = strategy.server.public.seal.Evaluator

    class AdditionSpy:
        def __init__(self, context):
            self.native = native_evaluator(context)
        def add_inplace(self, left, right):
            additions.append(1)
            self.native.add_inplace(left, right)

    monkeypatch.setattr(strategy.server.public.seal, 'Evaluator', AdditionSpy)
    original = strategy.decryptor.recover

    def counted(aggregate):
        calls.append(aggregate)
        return original(aggregate)

    monkeypatch.setattr(strategy.decryptor, 'recover', counted)
    result, metrics = aggregate_states(strategy, states, counts)
    plain = fedavg_aggregate(states, counts)
    for name in ('weight', 'bias', 'empty'):
        assert torch.allclose(result[name], plain[name], atol=1e-6, rtol=1e-6)
        assert result[name].dtype == state[name].dtype and result[name].shape == state[name].shape
    assert torch.equal(result['counter'], states[0]['counter'])
    assert len(calls) == 1 and len(calls[0].chunks) == 2
    assert len(additions) == 4  # two additions for each of two chunks
    assert metrics['he_ciphertext_count'] == 6
    assert metrics['he_ciphertext_upload_bytes'] > 0
    assert metrics['he_total_time_seconds'] == sum(value for key, value in metrics.items() if key.endswith('_time_seconds') and key != 'he_total_time_seconds')


def test_vector_exceeding_capacity_and_multi_round_recovery():
    state = {'large': torch.linspace(-1.0, 1.0, 8201), 'scalar': torch.tensor(-0.001)}
    strategy = strategy_for(state)
    for round_id in (1, 2):
        result, metrics = aggregate_states(strategy, [state], [3], round_id)
        assert torch.allclose(result['large'], state['large'], atol=1e-6, rtol=1e-6)
        assert torch.allclose(result['scalar'], state['scalar'], atol=1e-6)
        assert metrics['he_ciphertext_count'] == 3


@pytest.mark.parametrize('count', [0, -1, 1.5, True, 2**54])
def test_invalid_counts(state, count):
    strategy = strategy_for(state)
    with pytest.raises(ValueError, match='positive integers'):
        strategy.client_encoder.prepare(state, count, 0, 1)


@pytest.mark.parametrize('bad', ['shape', 'order', 'dtype', 'nan', 'inf'])
def test_invalid_models(state, bad):
    strategy = strategy_for(state)
    altered = copy.deepcopy(state)
    if bad == 'shape':
        altered['weight'] = altered['weight'].reshape(-1)
    elif bad == 'order':
        altered = dict(reversed(list(altered.items())))
    elif bad == 'dtype':
        altered['weight'] = altered['weight'].float()
    else:
        altered['weight'][0, 0] = float(bad)
    with pytest.raises(ValueError, match='schema|Nonfinite'):
        strategy.client_encoder.prepare(altered, 10, 0, 1)


def test_seal_security_constraints():
    # Structurally well-formed but exceeds SEAL's TC128 coefficient bound.
    with pytest.raises(ValueError, match='TC128'):
        create_context(ExperimentConfig(aggregation='ckks', ckks_poly_modulus_degree=4096))


def test_overflow_range_and_invalid_chunk_size(state):
    strategy = strategy_for(state)
    altered = copy.deepcopy(state)
    altered['weight'][:] = 1e100
    with pytest.raises(ValueError, match='encoding range'):
        strategy.client_encoder.prepare(altered, 1, 0, 1)
    with pytest.raises(ValueError, match='capacity'):
        CKKSEncoder(strategy.client_encoder.public, state_schema(state), 4097)


def test_server_rejects_wrong_message_types(state):
    strategy = strategy_for(state)
    with pytest.raises(ValueError, match='EncryptedUpdate'):
        strategy.server.aggregate([state])


def test_ciphertext_envelopes_tampering_and_authorization(state):
    strategy = strategy_for(state)
    update, _ = strategy.client_encoder.prepare(state, 20, 0, 1)
    assert isinstance(update, EncryptedUpdate) and all(isinstance(c, bytes) for c in update.chunks)
    assert not any(t.is_floating_point() for t in update.preserved.values())
    with pytest.raises(ValueError, match='never client updates'):
        strategy.decryptor.recover(update)
    for changed, match in [
        ({'context_id': 'another-key'}, 'context'),
        ({'chunk_lengths': (1,)}, 'chunk lengths'),
        ({'chunks': (b'corrupt ciphertext',)}, 'ciphertext'),
        ({'preserved': {}}, 'metadata'),
    ]:
        with pytest.raises(ValueError, match=match):
            strategy.server.aggregate([dataclasses.replace(update, **changed)])
    with pytest.raises(ValueError, match='Duplicate'):
        strategy.server.aggregate([update, update])
    aggregate, _ = strategy.server.aggregate([update])
    with pytest.raises(ValueError, match='Unauthorized'):
        strategy.decryptor.recover(aggregate)
    strategy.begin_round(1, ((0, 20),))
    recovered, _ = strategy.decryptor.recover(aggregate)
    assert torch.allclose(recovered['weight'], state['weight'], atol=1e-6)
    with pytest.raises(ValueError, match='Unauthorized'):
        strategy.decryptor.recover(aggregate)


def test_different_key_contexts_rejected(state):
    first, second = strategy_for(state), strategy_for(state)
    a, _ = first.client_encoder.prepare(state, 1, 0, 1)
    b, _ = second.client_encoder.prepare(state, 1, 1, 1)
    with pytest.raises(ValueError, match='context'):
        first.server.aggregate([a, b])


def test_nonfloating_only_model_and_bfloat16():
    for state in ({'n': torch.tensor(5), 'empty': torch.empty(0)}, {'v': torch.tensor([-0.5, 0.125], dtype=torch.bfloat16)}):
        strategy = strategy_for(state)
        result, _ = aggregate_states(strategy, [state], [1])
        for key in state:
            assert torch.equal(result[key], state[key])


def test_client_never_returns_plaintext_state_when_ckks(state):
    model = torch.nn.Linear(2, 2)
    strategy = strategy_for(model.state_dict())
    loader = DataLoader(TensorDataset(torch.ones(2, 2), torch.tensor([0, 1])), batch_size=2)
    result = FederatedClient(0, loader, model).local_train(1, 0.01, update_encoder=strategy.client_encoder, round_id=1)
    assert 'state_dict' not in result
    assert isinstance(result['encrypted_update'], EncryptedUpdate)


def test_training_partial_participation_checkpoints_and_finite_state(tmp_path):
    torch.manual_seed(42)
    data = TensorDataset(torch.randn(12, 2), torch.arange(12) % 3)

    class Factory:
        def get_labels(self):
            return data.tensors[1].numpy()
        def create_subset(self, indices):
            return DataLoader(Subset(data, indices), batch_size=2)

    config = ExperimentConfig(aggregation='ckks', num_clients=3, rounds=2, local_epochs=1,
                              client_fraction=2/3, optimizer='adam', device='cpu')
    model, history = run_federated_training(Factory(), DataLoader(data, batch_size=4), config,
                                            tmp_path, torch.nn.Linear(2, 3))
    assert len(history) == 2 and all(len(h['selected_client_ids']) == 2 for h in history)
    assert all(torch.isfinite(t).all() for t in model.state_dict().values())
    assert all(h['he_ciphertext_count'] == 2 for h in history)
    assert (tmp_path / 'experiment_config.json').exists()
    checkpoint = torch.load(tmp_path / 'checkpoint_round_2.pt', weights_only=False)
    assert set(checkpoint) == {'round', 'model_state', 'config'}
    assert not any('secret' in p.name.lower() or 'private' in p.name.lower() for p in tmp_path.rglob('*'))
