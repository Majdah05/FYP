from __future__ import annotations

import copy
import math
import numbers
import time

import numpy as np
import torch

from .ckks_context import deserialize, serialize_public_or_ciphertext
from .interface import EncryptedUpdate, HEMetrics, TensorSpec


def validate_sample_count(count):
    if isinstance(count, bool) or not isinstance(count, numbers.Integral) or not 0 < count <= 2 ** 53:
        raise ValueError('Sample counts must be positive integers no larger than 2**53.')


def state_schema(state):
    result = []
    for name, value in state.items():
        if not isinstance(name, str):
            raise ValueError('State dictionary keys must be strings.')
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.is_quantized:
                raise ValueError(f'Unsupported tensor layout for {name}.')
            result.append(TensorSpec(name, tuple(value.shape), str(value.dtype), value.is_floating_point(), value.numel()))
        else:
            result.append(TensorSpec(name, None, f'{type(value).__module__}.{type(value).__qualname__}', False, 0))
    return tuple(result)


def chunk_lengths(schema, chunk_size):
    total = sum(spec.numel for spec in schema if spec.floating)
    return tuple(min(chunk_size, total - start) for start in range(0, total, chunk_size))


def validate_preserved(schema, preserved):
    if set(preserved) != {spec.name for spec in schema if not spec.floating}:
        raise ValueError('Non-floating buffer metadata does not match the model schema.')
    for spec in schema:
        if spec.floating:
            continue
        value = preserved[spec.name]
        if state_schema({spec.name: value}) != (spec,):
            raise ValueError(f'Invalid preserved buffer: {spec.name}.')


def restore_state(schema, values, preserved):
    validate_preserved(schema, preserved)
    if values.size != sum(spec.numel for spec in schema if spec.floating) or not np.isfinite(values).all():
        raise ValueError('Recovered CKKS values have invalid length or contain nonfinite values.')
    state, offset = {}, 0
    for spec in schema:
        if not spec.floating:
            state[spec.name] = copy.deepcopy(preserved[spec.name])
            continue
        dtype = getattr(torch, spec.dtype.removeprefix('torch.'))
        tensor = torch.from_numpy(values[offset:offset + spec.numel].copy()).reshape(spec.shape).to(dtype=dtype)
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Recovered values overflow tensor dtype for {spec.name}.')
        state[spec.name] = tensor
        offset += spec.numel
    return state


def load_ciphertext(public, data, length):
    try:
        value = deserialize(public.seal.Ciphertext(), public.context, data)
        if value.size() != 2 or value.parms_id() != public.context.first_parms_id() or value.scale != public.scale:
            raise ValueError('Ciphertext has an incompatible level, scale, or size.')
        return value
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f'Invalid or incompatible CKKS ciphertext: {exc}') from exc


class CKKSEncoder:
    """Client-side packer with no secret key. Floating values cross as bytes."""
    def __init__(self, public, schema, chunk_size=None):
        self.public = public
        self.schema = schema
        self.chunk_size = chunk_size or public.slot_count
        self.participant_count = 1
        if not 0 < self.chunk_size <= public.slot_count:
            raise ValueError('CKKS chunk size exceeds packing capacity.')

    def prepare(self, state, sample_count, client_id, round_id):
        metrics = HEMetrics()
        start = time.perf_counter()
        validate_sample_count(sample_count)
        if state_schema(state) != self.schema:
            raise ValueError('Client model schema (names/order/shapes/dtypes) does not match the global model.')
        flat = np.empty(sum(spec.numel for spec in self.schema if spec.floating), dtype=np.float64)
        offset, preserved = 0, {}
        for spec in self.schema:
            value = state[spec.name]
            if not spec.floating:
                preserved[spec.name] = value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                continue
            # float64 conversion also supports torch.bfloat16, which NumPy cannot represent.
            array = value.detach().to(device='cpu', dtype=torch.float64).numpy().reshape(-1)
            if not np.isfinite(array).all():
                raise ValueError(f'Nonfinite client model values in {spec.name}.')
            with np.errstate(over='ignore', invalid='ignore'):
                flat[offset:offset + spec.numel] = array * sample_count
            offset += spec.numel
        if not np.isfinite(flat).all():
            raise ValueError('Sample weighting produced nonfinite CKKS values.')
        self.public.check_magnitude((float(np.max(np.abs(flat))) if flat.size else 0.0) * self.participant_count)
        lengths = chunk_lengths(self.schema, self.chunk_size)
        metrics.he_packaging_time_seconds = time.perf_counter() - start
        chunks, offset = [], 0
        start = time.perf_counter()
        encryptor = self.public.seal.Encryptor(self.public.context, self.public.public_key)
        metrics.he_encryption_time_seconds += time.perf_counter() - start
        for length in lengths:
            start = time.perf_counter()
            plain = self.public.seal.Plaintext()
            self.public.encoder.encode(flat[offset:offset + length].tolist(), self.public.scale, plain)
            encrypted = self.public.seal.Ciphertext()
            encryptor.encrypt(plain, encrypted)
            metrics.he_encryption_time_seconds += time.perf_counter() - start
            start = time.perf_counter()
            chunks.append(serialize_public_or_ciphertext(encrypted))
            metrics.he_serialization_time_seconds += time.perf_counter() - start
            offset += length
        return EncryptedUpdate(self.public.context_id, round_id, client_id, int(sample_count), self.schema,
                               tuple(chunks), lengths, preserved), metrics
