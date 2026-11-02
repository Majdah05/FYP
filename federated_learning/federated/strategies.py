"""Select aggregation without importing optional HE dependencies in plaintext."""
from __future__ import annotations

from .aggregation import fedavg_aggregate


class PlaintextStrategy:
    client_encoder = None
    metadata = {'aggregation': 'plaintext'}

    def begin_round(self, round_id, participants):
        pass

    def aggregate(self, updates, sample_counts, client_metrics):
        return fedavg_aggregate(updates, sample_counts), {}


class CKKSStrategy:
    """Trusted simulation coordinator; the server role itself has no secret."""
    def __init__(self, config, initial_state):
        from federated_learning.encryption.ckks_context import create_context
        from federated_learning.encryption.ckks_encoder import CKKSEncoder, state_schema
        from federated_learning.encryption.ckks_aggregation import CKKSAggregator

        public, self.decryptor, self.metadata = create_context(config)
        schema = state_schema(initial_state)
        self.client_encoder = CKKSEncoder(public, schema, config.ckks_chunk_size)
        self.server = CKKSAggregator(public, schema, self.client_encoder.chunk_size)
        self.decryptor.bind_schema(schema, self.server.lengths)
        self.metadata['aggregation'] = 'ckks'
        self._expected = ()

    def begin_round(self, round_id, participants):
        self._expected = tuple(participants)
        self.client_encoder.participant_count = len(self._expected)
        self.decryptor.authorize(round_id, self._expected)

    def aggregate(self, updates, sample_counts, client_metrics):
        from federated_learning.encryption.interface import HEMetrics

        if tuple((u.client_id, u.sample_count) for u in updates) != self._expected:
            raise ValueError('Encrypted uploads do not match selected clients and dataset sizes.')
        if list(sample_counts) != [u.sample_count for u in updates]:
            raise ValueError('Sample counts do not match encrypted contributions.')
        metrics = HEMetrics()
        for item in client_metrics:
            metrics.add(item)
        aggregate, server_metrics = self.server.aggregate(updates)
        state, recovery_metrics = self.decryptor.recover(aggregate)
        metrics.add(server_metrics)
        metrics.add(recovery_metrics)
        result = metrics.as_dict()
        result.update({
            'he_ciphertext_upload_bytes': sum(len(c) for u in updates for c in u.chunks),
            'he_aggregate_ciphertext_bytes': sum(map(len, aggregate.chunks)),
            'he_ciphertext_count': sum(len(u.chunks) for u in updates),
            'he_aggregate_ciphertext_count': len(aggregate.chunks),
            'he_public_context_bytes': self.metadata['he_public_context_bytes'],
            'he_traffic_basis': self.metadata['traffic_basis'],
        })
        return state, result


def create_aggregation_strategy(config, initial_state):
    mode = getattr(config, 'aggregation', 'plaintext')
    if mode == 'plaintext':
        return PlaintextStrategy()
    if mode == 'ckks':
        if config.he_backend != 'tenseal_seal':
            raise ValueError('CKKS requires he_backend=tenseal_seal.')
        return CKKSStrategy(config, initial_state)
    raise ValueError(f'Unsupported aggregation mode: {mode}')
