import builtins
import sys

import pytest

from federated_learning.config import ExperimentConfig, parse_config
from federated_learning.federated.strategies import create_aggregation_strategy


def test_yaml_is_preserved_and_explicit_cli_overrides(tmp_path):
    path = tmp_path / 'experiment.yaml'
    path.write_text('rounds: 7\nnum_clients: 3\naggregation: ckks\nckks_coeff_mod_bit_sizes: [60, 40, 60]\nckks_chunk_size: 256\nsave_checkpoints: false\nsmoke_test: true\n')
    config = parse_config(['--config', str(path), '--rounds', '2', '--no-smoke_test'])
    assert (config.rounds, config.num_clients, config.aggregation) == (2, 3, 'ckks')
    assert config.ckks_coeff_mod_bit_sizes == [60, 40, 60]
    assert config.ckks_chunk_size == 256
    assert config.save_checkpoints is False and config.smoke_test is False


def test_cli_defaults_and_ckks_list():
    assert parse_config([]) == ExperimentConfig()
    config = parse_config(['--aggregation', 'ckks', '--ckks_coeff_mod_bit_sizes', '60', '40', '60'])
    assert config.ckks_coeff_mod_bit_sizes == [60, 40, 60]


@pytest.mark.parametrize('field,value,match', [
    ('aggregation', 'fake', 'aggregation'), ('he_backend', 'fake', 'he_backend'),
    ('ckks_poly_modulus_degree', 1234, 'power of two'),
    ('ckks_poly_modulus_degree', True, 'power of two'),
    ('ckks_coeff_mod_bit_sizes', '60,40', 'at least two'),
    ('ckks_coeff_mod_bit_sizes', [60, 61], 'at least two'),
    ('ckks_coeff_mod_bit_sizes', [True, 40], 'at least two'),
    ('ckks_global_scale_power', 140, 'scale_power'),
    ('ckks_chunk_size', 4097, 'N/2'), ('ckks_chunk_size', 0, 'N/2'),
])
def test_bad_he_config(field, value, match):
    with pytest.raises(ValueError, match=match):
        ExperimentConfig(**{field: value}).validate()


def test_bad_yaml(tmp_path):
    path = tmp_path / 'bad.yaml'
    path.write_text('- not\n- a mapping\n')
    with pytest.raises(ValueError, match='mapping'):
        parse_config(['--config', str(path)])
    path.write_text('[]')
    with pytest.raises(ValueError, match='mapping'):
        parse_config(['--config', str(path)])
    path.write_text('unknown_setting: 1')
    with pytest.raises(ValueError, match='Unknown'):
        parse_config(['--config', str(path)])


def test_optional_dependency_is_lazy_and_missing_ckks_fails(monkeypatch):
    original_import = builtins.__import__
    attempted = []

    def guarded(name, *args, **kwargs):
        if name == 'tenseal' or name.startswith('tenseal.'):
            attempted.append(name)
            raise ImportError('deliberately unavailable for regression test')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    strategy = create_aggregation_strategy(parse_config([]), {})
    assert strategy.client_encoder is None and not attempted
    with pytest.raises(RuntimeError, match='requirements-he.txt'):
        create_aggregation_strategy(ExperimentConfig(aggregation='ckks'), {})
    assert attempted
