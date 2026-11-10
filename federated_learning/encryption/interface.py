"""Public role boundaries; ciphertext messages never contain floating updates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Any


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...] | None
    dtype: str
    floating: bool
    numel: int


@dataclass
class HEMetrics:
    he_packaging_time_seconds: float = 0.0
    he_encryption_time_seconds: float = 0.0
    he_aggregation_time_seconds: float = 0.0
    he_decryption_time_seconds: float = 0.0
    he_serialization_time_seconds: float = 0.0

    def add(self, other: HEMetrics):
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self):
        result = vars(self).copy()
        result['he_total_time_seconds'] = sum(result.values())
        return result


@dataclass(frozen=True)
class EncryptedUpdate:
    context_id: str
    round_id: int
    client_id: int
    sample_count: int
    schema: tuple[TensorSpec, ...]
    chunks: tuple[bytes, ...]
    chunk_lengths: tuple[int, ...]
    preserved: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class EncryptedAggregate:
    context_id: str
    round_id: int
    participants: tuple[tuple[int, int], ...]
    schema: tuple[TensorSpec, ...]
    chunks: tuple[bytes, ...]
    chunk_lengths: tuple[int, ...]
    preserved: dict[str, Any] = field(repr=False)


class ClientUpdateEncoder(Protocol):
    def prepare(self, state: dict, sample_count: int, client_id: int,
                round_id: int) -> tuple[EncryptedUpdate, HEMetrics]: ...


class AggregateDecryptor(Protocol):
    """Replace this role for future collaborative decryption, not local training."""
    def authorize(self, round_id: int, participants: tuple[tuple[int, int], ...]): ...
    def recover(self, aggregate: EncryptedAggregate) -> tuple[dict, HEMetrics]: ...
