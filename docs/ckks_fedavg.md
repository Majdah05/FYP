# CKKS encrypted FedAvg

This project can aggregate locally trained PyTorch models with Microsoft SEAL
CKKS. Local training, inference, validation, optimizer state and the released
global model remain plaintext. This is encrypted model aggregation, not
homomorphic neural-network training.

## Installation

Keep the baseline installation unchanged:

```bash
python -m pip install -r requirements.txt
```

For CKKS, install the optional interface:

```bash
python -m pip install -r requirements-he.txt
python -c "import tenseal; import tenseal.sealapi; print(tenseal.__version__)"
```

The verified pin is TenSEAL 0.3.18. A `cp314-cp314-win_amd64` wheel was installed
and exercised on Windows 11 with Python 3.14.0. TenSEAL is a Python/C++ interface
to Microsoft SEAL; Microsoft's C++ API itself is not a native Python package.
The implementation uses TenSEAL's documented `tenseal.sealapi` bindings because
the high-level `ts.context` eagerly generates relinearization keys. Only a
public/secret key pair is generated here; there are no multiplication, rotation,
relinearization or bootstrapping operations or keys.

Use a virtual environment on Windows (`python -m venv .venv`, then
`.venv\Scripts\Activate.ps1`) or Linux (`python -m venv .venv`, then
`source .venv/bin/activate`). Install both requirements files inside it.
In Colab, run `%pip install -r requirements.txt -r requirements-he.txt` in the
checked-out repository, then use the same commands below. CUDA helps PyTorch
training; SEAL operations run on CPU.

Wheel availability depends on Python version, CPU architecture and operating
system. Verify it with `python -m pip download --only-binary=:all: --no-deps
tenseal==0.3.18`. If no matching wheel is available, use a supported interpreter
in an isolated environment or follow the official TenSEAL source-build process.
That requires a C++17 compiler and CMake; Windows additionally needs Visual
Studio C++ build tools. Source builds fetch dependencies and need network access.
Consult the release's instructions for CMake compatibility. The smallest fallback
would be an official SEAL C++ build plus a narrow Python binding implementing
these role interfaces; it is not implemented or silently substituted here.

Build provenance matters: the
[TenSEAL v0.3.18 build manifest](https://github.com/OpenMined/TenSEAL/blob/v0.3.18/cmake/seal.cmake)
pins SEAL v4.3.3. The installed binding does not expose a runtime SEAL version,
so metadata distinguishes the manifest reference from runtime verification.
Microsoft's current [SEAL guidance](https://github.com/microsoft/SEAL) identifies
4.4.0+ as a security update for untrusted-input handling. This verified wheel is
therefore suitable here for controlled research with locally generated messages;
upgrade and revalidate the underlying SEAL build before handling adversarial
serialized inputs or deploying a service.

## Commands and configuration

The default is plaintext and does not import TenSEAL or create a SEAL context:

```bash
python main.py --dataset pathmnist --smoke_test --num_clients 3 --rounds 1 --local_epochs 1 --batch_size 32 --lr 0.01
python main.py --dataset pathmnist --aggregation plaintext --smoke_test --num_clients 3 --rounds 1 --local_epochs 1 --batch_size 32 --lr 0.01
python main.py --dataset pathmnist --aggregation ckks --smoke_test --num_clients 3 --rounds 1 --local_epochs 1 --batch_size 32 --lr 0.01
python main.py --config configs/pathmnist_iid_ckks.yaml
```

The original `main.py` also runs a separate centralized baseline on the FULL
official training split, even with `--smoke_test`. The federated smoke workload
uses the first 64 training examples; validation and held-out test use official
splits. Allow time for the centralized stage. The paired harness below exercises
the same federated loop without rerunning that separate baseline.

New settings are available as YAML keys or CLI flags with the same names:

| Setting | Default | Meaning |
| --- | --- | --- |
| `aggregation` | `plaintext` | `plaintext` or `ckks` |
| `he_backend` | `tenseal_seal` | supported CKKS backend |
| `ckks_poly_modulus_degree` | `8192` | polynomial degree N |
| `ckks_coeff_mod_bit_sizes` | `[60, 40, 40, 60]` | coefficient-modulus prime bit sizes |
| `ckks_global_scale_power` | `40` | encoding scale is 2 to this power |
| `ckks_chunk_size` | omitted | defaults to N/2; cannot exceed it |

On CLI, use `--ckks_coeff_mod_bit_sizes 60 40 40 60`. Explicit CLI arguments
override YAML; absent arguments retain YAML or dataclass defaults. This fixes
the previous parser-default overwrite bug. Boolean settings can be disabled
explicitly with `--no-save_checkpoints` and `--no-smoke_test`; the declared
dataclass checkpoint default is now consistently used when neither source sets it.
Original YAML files are retained. The four new paired configurations differ only
in aggregation settings within each distribution.

Structural parameter checks run without TenSEAL. Each actual CKKS context is
constructed with `SEC_LEVEL_TYPE.TC128` and must pass SEAL's parameter checks.
There is no `SEC_LEVEL_TYPE.NONE` bypass. The initial 200-bit modulus at degree
8192 passes the library's 128-bit security constraints. This validates encryption
parameters, not the security of an entire federated protocol.
Invalid parameters, scale, chunk sizes or missing dependencies fail explicitly;
there is no fallback to plaintext.

## Mathematics and packing

Client k encrypts `n_k * w_k`, where `n_k` is its number of local samples.
The server adds matching ciphertext chunks. The trusted decryptor decodes only
the sum and divides by the total number of selected samples:

```text
client k:       c_k = Enc(n_k * w_k)
server:         C = sum_k c_k             (SEAL Evaluator.add_inplace)
decryptor:      w_new ≈ Dec(C) / sum_k n_k
```

Only participating clients contribute to the denominator. Counts must be positive
integers and the total must fit exact float64 integer representation (2**53).
CKKS supports approximate real arithmetic and efficient packed vectors, making it
suitable for floating model weights. It does not provide exact arithmetic.

`CKKSEncoder` walks state tensors in state-dictionary order. It records names,
shapes and dtypes, converts floating values to a CPU float64 flat array, weights
them, and splits the array across tensors into chunks of at most N/2 slots.
At degree 8192 the capacity is 4096. The entire CNN is covered; tensor boundaries
do not require separate ciphertexts. Empty floating tensors retain their shape
and require no slots. Reconstruction restores original dtypes on CPU; the
existing `load_state_dict` copies tensors to the global model's CPU/CUDA device.

Names, order, shapes, dtypes and chunk lengths must match the initial model.
Nonfinite values and conservative encoding-range violations are rejected before
encryption. Sparse and quantized states are rejected explicitly. Non-floating
buffers (including integer counters) and non-tensor extra state follow the
existing baseline policy: copy the first participating client's value, in
plaintext. Such metadata is NOT protected by CKKS and must not contain secrets.
Empty/all-non-floating models are supported. Float64 packing avoids intermediate
float16/bfloat16 weighting overflow; final dtype conversion is checked for finite
values. Numerical tolerances must be remeasured after changing scale, magnitude,
dtype, modulus or client count.

## Roles and authorization

`create_context()` generates one compatible key pair per experiment. Only public
parameters and the public key are serialized into `PublicContextBundle`.
`PublicContext` loads that bundle with no secret key. The key generator is local
to initialization, and its secret is retained only inside the decryptor primitive.

`FederatedClient.local_train()` optionally packages its updated model before
returning. Its CKKS result contains an `EncryptedUpdate` with byte chunks, public
schema, sample count, client/round/context IDs and preserved non-floating metadata;
it does not return a floating plaintext state dictionary. `CKKSAggregator` has
only a public context. It validates compatible envelopes and invokes SEAL
ciphertext addition. `TrustedDecryptor` accepts `EncryptedAggregate`, rejects
client messages, requires coordinator authorization for the expected round and
participant/sample-count roster, and consumes that authorization once.

These are logical roles in one Python process. They are not memory isolation,
cryptographic authentication or proof that a malicious server honestly added
the declared roster. A server with process memory access can access local states
and secrets. Context IDs detect ordinary key-context mistakes; they are not
signatures. No standard SEAL API can detect a ciphertext encrypted under another
key with the same parameters if an adversary forges its envelope. No network or
hostile-input service is provided.

Assumptions: an honest coordinator/decryptor, controlled participants and correct
metadata. CKKS conceals individual updates from a separate server without the
secret key. Released aggregates can still leak information; one selected client
reveals its update through the aggregate. Server manipulation, model poisoning,
collusion with the decryptor, replay across independently deployed services,
traffic analysis and aggregate inference are not solved by this implementation.
The authorization checks are defensive routing checks, not a malicious-secure
aggregation protocol.

`AggregateDecryptor` is the replacement point for a future collaborative recovery
service. That would need a real threshold-capable cryptosystem/protocol,
distributed key generation, authentication and a release policy. Standard
TenSEAL does not provide threshold decryption here. No zero-order optimization,
differential privacy or threshold mechanism has been added.

## Measurement and results

Every CKKS round extends the existing CSV/JSON history with these measurements:

| Metric | Measurement |
| --- | --- |
| `he_packaging_time_seconds` | validation, CPU conversion, flattening and weighting |
| `he_encryption_time_seconds` | public encryptor creation, encoding and encryption |
| `he_aggregation_time_seconds` | envelope validation and ciphertext addition |
| `he_decryption_time_seconds` | authorization checks, aggregate decryption, decoding, averaging and reconstruction |
| `he_serialization_time_seconds` | ciphertext save/load, including temporary-file I/O |
| `he_total_time_seconds` | sum of these non-overlapping measured intervals |
| `he_ciphertext_upload_bytes` | sum of actual serialized client ciphertext lengths |
| `he_aggregate_ciphertext_bytes` | actual serialized aggregate ciphertext lengths |
| `he_ciphertext_count` | total uploaded ciphertext count |
| `he_aggregate_ciphertext_count` | aggregate ciphertext count |
| `he_public_context_bytes` | serialized public parameters plus public-key size; reported separately, not retransmitted every round |

All timings use `time.perf_counter()`. These interval sums exclude uninstrumented
Python dispatch overhead; the offline benchmark also measures pipeline wall time.
Public-context serialization and key generation are included only in
`he_initialization_time_seconds` in `experiment_config.json`, not in round totals.
Public context is reused in-process; no physical transmission occurs.

Byte counts are **simulated serialized SEAL payload traffic**, not measured
network traffic. They exclude metadata, transport framing, global-model download,
and returned plaintext aggregate. SEAL serialization includes its headers and
compression. The low-level bindings use temporary files containing public data
or ciphertext only; no private key or plaintext update is saved there.

`communication_volume_bytes` retains the original hypothetical bidirectional
plaintext model estimate, even in CKKS mode, with an explicit basis label.
`plaintext_float_upload_bytes_estimate` counts native floating tensor storage for
comparison to ciphertext upload; it is an estimate, not a wire measurement.
Compare these two upload metrics, not ciphertext upload to the legacy
bidirectional estimate. Non-floating metadata costs are excluded from that plot.

Existing training loss, validation metrics, macro F1, held-out test, checkpoints
and plots remain available. `experiment_config.json` includes effective crypto
parameters, public-context size, package versions and federated wall time.
`execution_summary.json` from `main.py` measures the full run including the
centralized baseline. Checkpoints contain only round, model state and configuration.
Keys, ciphertext objects and trusted-decryptor instances are never persisted.

## Reproducible comparisons

From the repository root, choose a NEW output directory for each paired run:

```bash
python -m scripts.compare_he --config configs/pathmnist_iid_ckks.yaml --output results/paired_iid
python -m scripts.compare_he --config configs/pathmnist_non_iid_ckks.yaml --output results/paired_non_iid
python -m scripts.compare_he --config configs/pathmnist_iid_ckks.yaml --smoke_test --rounds 2 --device cpu --output results/paired_smoke
python -m scripts.benchmark_ckks --full_model --client_counts 2 3 5 --repeats 3
```

The paired harness resets PyTorch/NumPy seeds before each mode, reuses identical
data and sample indices, records a partition hash, and verifies equal selected
client schedules. Models, optimizer, hyperparameters and starting RNG state are
matched. GPU scheduling can still be nondeterministic; use CPU for controlled
comparisons. It generates accuracy, validation loss, round time, crypto time and
communication plots plus held-out test metrics. Two-round smoke curves establish
functionality only; they do not establish long-run convergence or clinical utility.

The offline benchmark uses the EXACT SAME synthetic client model states and
unequal counts in both aggregation modes. It reports maximum absolute error,
RMSE, relative error with a 1e-6 denominator floor, timings and bytes, and plots
error and client-count overhead. No plaintext precision reference is computed in
the normal encrypted training path. Native float32 FedAvg rounding also
contributes to its difference from a float64-packed CKKS result.

## Tests

```bash
python -m pytest tests -q
python -m pytest tests -q -m he
python -m pytest tests -q -m ""
```

HE tests are excluded by default and skip when TenSEAL is unavailable. The first
command runs plaintext/configuration regression tests without HE dependencies.
The second enables real SEAL encryption, key separation, addition, serialization,
chunking, numerical and input checks, aggregate-only recovery, partial
participation and multiround training. The third enables every test. In restricted
Windows environments, provide a writable new `--basetemp .test-run` directory if
the OS temp directory is inaccessible. CKKS operations use the standard writable
temporary directory; adjust `TEMP`/`TMP` if required by the execution environment.

See [the measured validation report](ckks_validation_report.md) for actual runs,
errors, overhead, file inventory and limitations.

References: [TenSEAL](https://github.com/OpenMined/TenSEAL),
[TenSEAL SEAL API](https://github.com/OpenMined/TenSEAL/tree/v0.3.18/tenseal/sealapi),
[Microsoft CKKS example](https://github.com/microsoft/SEAL/blob/main/native/examples/5_ckks_basics.cpp).
