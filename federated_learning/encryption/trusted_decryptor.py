"""Trusted recovery service. This is not threshold decryption or isolation."""
from __future__ import annotations

import time

import numpy as np

from .ckks_encoder import load_ciphertext, restore_state, validate_sample_count
from .interface import EncryptedAggregate, HEMetrics


class TrustedDecryptor:
    def __init__(self, public, secret_decryptor):
        self.public = public
        self.__decryptor = secret_decryptor
        self.__authorized = {}
        self.__used = set()
        self.schema = None
        self.lengths = None

    def bind_schema(self, schema, lengths):
        if self.schema is not None:
            raise ValueError('Decryptor schema is already bound.')
        self.schema, self.lengths = schema, lengths

    def authorize(self, round_id, participants):
        """Called by the trusted coordinator, never by the aggregation role.

        This checks routing/roster/replay errors, not malicious-server correctness.
        No cryptographic proof authenticates that a sum contains this roster.
        """
        if round_id in self.__used or round_id in self.__authorized:
            raise ValueError('Aggregate round was already authorized or recovered.')
        if not participants or len({client for client, _ in participants}) != len(participants):
            raise ValueError('Aggregate authorization needs a nonempty unique participant roster.')
        for _, count in participants:
            validate_sample_count(count)
        self.__authorized[round_id] = tuple(participants)

    def recover(self, aggregate):
        start = time.perf_counter()
        if type(aggregate) is not EncryptedAggregate:
            raise ValueError('Trusted decryptor only accepts authorized aggregate messages, never client updates.')
        expected = self.__authorized.get(aggregate.round_id)
        if expected is None or expected != aggregate.participants or aggregate.context_id != self.public.context_id:
            raise ValueError('Unauthorized aggregate context, round, or participant roster.')
        if aggregate.schema != self.schema or aggregate.chunk_lengths != self.lengths or len(aggregate.chunks) != len(self.lengths):
            raise ValueError('Aggregate schema or chunk lengths do not match the authorized model.')
        total_samples = sum(count for _, count in expected)
        if total_samples > 2 ** 53:
            raise ValueError('Total sample count exceeds exact float64 integer range.')
        # Consume authorization even if a later native load/decryption fails.
        self.__authorized.pop(aggregate.round_id)
        self.__used.add(aggregate.round_id)
        metrics = HEMetrics(he_decryption_time_seconds=time.perf_counter() - start)
        values = []
        for data, length in zip(aggregate.chunks, self.lengths):
            start = time.perf_counter()
            encrypted = load_ciphertext(self.public, data, length)
            metrics.he_serialization_time_seconds += time.perf_counter() - start
            start = time.perf_counter()
            plain = self.public.seal.Plaintext()
            self.__decryptor.decrypt(encrypted, plain)
            values.extend(self.public.encoder.decode_double(plain)[:length])
            metrics.he_decryption_time_seconds += time.perf_counter() - start
        start = time.perf_counter()
        state = restore_state(self.schema, np.asarray(values, dtype=np.float64) / total_samples, aggregate.preserved)
        metrics.he_decryption_time_seconds += time.perf_counter() - start
        return state, metrics
