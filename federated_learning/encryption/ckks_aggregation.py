"""Public-only server: validate envelopes and add matching ciphertext chunks."""
from __future__ import annotations

import copy
import time

from .ckks_context import serialize_public_or_ciphertext
from .ckks_encoder import chunk_lengths, load_ciphertext, validate_preserved, validate_sample_count
from .interface import EncryptedAggregate, EncryptedUpdate, HEMetrics


class CKKSAggregator:
    def __init__(self, public, schema, chunk_size):
        self.public = public
        self.schema = schema
        self.lengths = chunk_lengths(schema, chunk_size)

    def aggregate(self, updates):
        start = time.perf_counter()
        if not updates:
            raise ValueError('At least one encrypted client contribution is required.')
        if any(type(update) is not EncryptedUpdate for update in updates):
            raise ValueError('The CKKS server accepts EncryptedUpdate messages only.')
        seen, round_id = set(), updates[0].round_id
        for update in updates:
            if type(update) is not EncryptedUpdate:
                raise ValueError('The CKKS server accepts EncryptedUpdate messages only.')
            validate_sample_count(update.sample_count)
            if update.context_id != self.public.context_id or update.round_id != round_id:
                raise ValueError('Client ciphertext context or round mismatch.')
            if update.client_id in seen:
                raise ValueError('Duplicate encrypted client contribution.')
            seen.add(update.client_id)
            if update.schema != self.schema or update.chunk_lengths != self.lengths or len(update.chunks) != len(self.lengths):
                raise ValueError('Client model schema or ciphertext chunk lengths do not match.')
            validate_preserved(self.schema, update.preserved)
        if sum(update.sample_count for update in updates) > 2 ** 53:
            raise ValueError('Total sample count exceeds exact float64 integer range.')
        evaluator = self.public.seal.Evaluator(self.public.context)
        metrics = HEMetrics(he_aggregation_time_seconds=time.perf_counter() - start)
        chunks = []
        for index, length in enumerate(self.lengths):
            total = None
            for update in updates:
                start = time.perf_counter()
                value = load_ciphertext(self.public, update.chunks[index], length)
                metrics.he_serialization_time_seconds += time.perf_counter() - start
                start = time.perf_counter()
                if total is None:
                    total = value
                else:
                    evaluator.add_inplace(total, value)
                metrics.he_aggregation_time_seconds += time.perf_counter() - start
            start = time.perf_counter()
            chunks.append(serialize_public_or_ciphertext(total))
            metrics.he_serialization_time_seconds += time.perf_counter() - start
        return EncryptedAggregate(self.public.context_id, round_id,
                                  tuple((u.client_id, u.sample_count) for u in updates), self.schema,
                                  tuple(chunks), self.lengths, copy.deepcopy(updates[0].preserved)), metrics
