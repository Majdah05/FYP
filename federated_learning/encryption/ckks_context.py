"""Microsoft SEAL via TenSEAL's documented low-level Python bindings.

The high-level ts.context eagerly generates relinearization keys. sealapi lets
this addition-only backend generate exactly one public/secret key pair instead.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


def load_seal():
    try:
        from tenseal import sealapi
    except ImportError as exc:
        raise RuntimeError(
            'CKKS requires TenSEAL with SEAL API bindings. Install with '
            '`python -m pip install -r requirements-he.txt`. No plaintext fallback.'
        ) from exc
    return sealapi


def validate_parameters(degree, bits, power, chunk_size):
    if type(degree) is not int or degree < 1024 or degree > 32768 or degree & (degree - 1):
        raise ValueError('ckks_poly_modulus_degree must be a power of two in [1024, 32768].')
    if not isinstance(bits, (list, tuple)) or len(bits) < 2 or any(
        type(b) is not int or not 2 <= b <= 60 for b in bits
    ):
        raise ValueError('ckks_coeff_mod_bit_sizes must contain at least two integers in [2, 60].')
    if type(power) is not int or not 1 <= power < min(1023, sum(bits[:-1])):
        raise ValueError('ckks_global_scale_power must be positive and below the initial data modulus bits.')
    if chunk_size is not None and (type(chunk_size) is not int or not 1 <= chunk_size <= degree // 2):
        raise ValueError('ckks_chunk_size must be a positive integer no larger than N/2 slots.')


def serialize_public_or_ciphertext(value) -> bytes:
    """sealapi exposes path-based save/load. Never call this with a secret key.

    Temporary files contain ONLY public context components or ciphertexts and
    are removed on every exit. File I/O is included in serialization timings.
    """
    with tempfile.TemporaryDirectory(prefix='fedavg-seal-') as folder:
        path = Path(folder) / 'payload.seal'
        value.save(str(path))
        return path.read_bytes()


def deserialize(value, context, data: bytes):
    if not isinstance(data, bytes) or not data:
        raise ValueError('Expected a nonempty serialized SEAL byte payload.')
    with tempfile.TemporaryDirectory(prefix='fedavg-seal-') as folder:
        path = Path(folder) / 'payload.seal'
        path.write_bytes(data)
        if context is None:
            value.load(str(path))
        else:
            value.load(context, str(path))
    return value


@dataclass(frozen=True)
class PublicContextBundle:
    parameters: bytes
    public_key: bytes
    scale_power: int

    @property
    def context_id(self):
        return hashlib.sha256(self.parameters + self.public_key + str(self.scale_power).encode()).hexdigest()

    @property
    def byte_count(self):
        # Raw serialized SEAL components only, excluding transport framing.
        return len(self.parameters) + len(self.public_key)


class PublicContext:
    """Contains SEAL parameters and public key only; cannot decrypt."""
    def __init__(self, bundle: PublicContextBundle):
        self.seal = load_seal()
        self.bundle = bundle
        self.context_id = bundle.context_id
        parms = deserialize(self.seal.EncryptionParameters(self.seal.SCHEME_TYPE.CKKS), None, bundle.parameters)
        self.context = self.seal.SEALContext(parms, True, self.seal.SEC_LEVEL_TYPE.TC128)
        if not self.context.parameters_set():
            raise ValueError(f'Invalid CKKS public context: {self.context.parameters_error_message()}')
        self.public_key = deserialize(self.seal.PublicKey(), self.context, bundle.public_key)
        self.encoder = self.seal.CKKSEncoder(self.context)
        self.scale = float(2 ** bundle.scale_power)
        self.slot_count = self.encoder.slot_count()
        self.data_modulus_bits = self.context.first_context_data().total_coeff_modulus_bit_count()
        if bundle.scale_power >= self.data_modulus_bits:
            raise ValueError('CKKS scale exceeds the validated SEAL data modulus.')

    def check_magnitude(self, maximum: float):
        # Conservative bound includes FFT encoding growth and two safety bits.
        if maximum and math.log2(maximum) + self.bundle.scale_power + math.log2(self.slot_count) + 2 >= self.data_modulus_bits:
            raise ValueError('Weighted CKKS values exceed the conservative encoding range; reduce counts/values or scale, or increase the data modulus securely.')


def create_context(config):
    """Generate once per experiment. Return public-only roles and secret owner."""
    from .trusted_decryptor import TrustedDecryptor

    start = time.perf_counter()
    degree, bits, power = config.ckks_poly_modulus_degree, config.ckks_coeff_mod_bit_sizes, config.ckks_global_scale_power
    validate_parameters(degree, bits, power, config.ckks_chunk_size)
    seal = load_seal()
    try:
        parms = seal.EncryptionParameters(seal.SCHEME_TYPE.CKKS)
        parms.set_poly_modulus_degree(degree)
        parms.set_coeff_modulus(seal.CoeffModulus.Create(degree, list(bits)))
        context = seal.SEALContext(parms, True, seal.SEC_LEVEL_TYPE.TC128)
        if not context.parameters_set():
            raise ValueError(context.parameters_error_message())
        keygen = seal.KeyGenerator(context)
        public_key = seal.PublicKey()
        keygen.create_public_key(public_key)
        bundle = PublicContextBundle(serialize_public_or_ciphertext(parms), serialize_public_or_ciphertext(public_key), power)
        public = PublicContext(bundle)
        decryptor = TrustedDecryptor(public, seal.Decryptor(context, keygen.secret_key()))
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f'CKKS initialization failed (SEAL TC128 validation): {exc}') from exc
    versions = {}
    for package in ('tenseal', 'torch', 'numpy', 'medmnist'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    metadata = {
        'he_backend': 'tenseal_seal', 'api': 'tenseal.sealapi',
        'poly_modulus_degree': degree, 'coeff_mod_bit_sizes': list(bits),
        'global_scale_power': power, 'chunk_size': config.ckks_chunk_size or degree // 2,
        'packing_capacity': public.slot_count, 'security_validation': 'SEAL SEC_LEVEL_TYPE.TC128',
        'he_public_context_bytes': bundle.byte_count,
        'he_initialization_time_seconds': time.perf_counter() - start,
        'package_versions': versions, 'python': platform.python_version(), 'platform': platform.platform(),
        'seal_runtime_version': 'not exposed by the installed TenSEAL Python binding',
        'seal_build_reference': 'TenSEAL v0.3.18 cmake/seal.cmake pins Microsoft SEAL v4.3.3',
        'traffic_basis': 'simulated serialized SEAL payload traffic; no network transport',
        'serialization': 'SEAL path-based save/load; temporary public/ciphertext I/O included',
        'relinearization_keys_generated': False, 'galois_keys_generated': False,
    }
    return public, decryptor, metadata
