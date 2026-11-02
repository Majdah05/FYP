# CKKS integration: measured validation report

Date: 2026-11-02. Target repository: `C:\Users\ritaj\OneDrive - American University of Beirut\Fall 2027\FYPP\FYP`.
Feature branch: `codex/ckks-fedavg`. The original checkout was clean before implementation. Changes are left uncommitted for review; no push was performed.

## Implementation outcome

CKKS is an optional aggregation strategy in the existing FedAvg loop. Clients train normally, pack their sample-weighted floating model states into actual Microsoft SEAL ciphertexts, and return serialized ciphertext payloads. The public-only aggregation role invokes `Evaluator.add_inplace`; the trusted decryptor recovers one authorized aggregate per round. The original `fedavg_aggregate()` source is unchanged and remains the plaintext strategy.

The entire PathMNIST CNN is covered: **241,993 floating values**, packed into **60 ciphertexts per client** with 4096 real slots each. Three participating clients upload 180 ciphertexts; the trusted decryptor receives 60 aggregate ciphertexts. No layers are omitted. Integer/non-floating buffers retain the first participating client's value, as in the baseline.

The configuration merge also now preserves YAML values unless a CLI flag is explicitly provided. Optional dependencies are loaded only for CKKS. Effective parameters, package versions, separate initialization costs, round timing and serialized payload sizes are recorded with existing experiment outputs.

## Environment and parameters

- Windows 11 x64; Python 3.14.0; PyTorch 2.14.1; NumPy 2.4.2; MedMNIST 3.0.2.
- TenSEAL 0.3.18 installed successfully from the `cp314-cp314-win_amd64` wheel.
- No CUDA device/build available in this environment; CPU training and HE were exercised. GPU loading is supported in code but was not tested here.
- Degree 8192; coefficient bits `[60, 40, 40, 60]`; scale `2**40`; chunk size 4096.
- Contexts pass SEAL `SEC_LEVEL_TYPE.TC128` checks. No relinearization/Galois keys are generated.
- The [TenSEAL release build manifest](https://github.com/OpenMined/TenSEAL/blob/v0.3.18/cmake/seal.cmake) pins SEAL 4.3.3; runtime SEAL version is not exposed by these bindings. See the security limitations below.
- `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2` were used for training/benchmarks. Some validation processes overlapped. These are exploratory local measurements, not isolated hardware performance benchmarks.

## Validation

- Original suite: **13 passed** before modification.
- Plaintext/configuration suite without TenSEAL on the import path: **27 passed**. It also explicitly blocks TenSEAL imports to verify lazy loading and a useful missing-dependency error, with no plaintext fallback.
- Full suite including actual SEAL operations in the target checkout: **48 passed**. Coverage includes negative/decimal/zero values, weighted averaging, float64/float32/bfloat16, empty and multi-chunk tensors, schema/count/magnitude checks, TC128 rejection, ciphertext serialization and addition, public-only roles, aggregate-only authorization, replay checks, partial participation, Adam and multiround recovery.
- A spy confirms four actual native ciphertext additions for three clients and two chunks, while the recovery service is called once for the aggregate.
- `git diff --check` passed.
- An initial restricted-environment baseline attempt had **11 passed and 2 fixture setup errors** because pytest could not create directories in the OS temporary folder. Selecting a writable `--basetemp` resolved those errors; there are no remaining test failures.

The original encrypted main command completed with 3 clients, 1 round, 1 local epoch, batch size 32 and the first 64 training examples. It produced full-model finite checkpoints, official validation/held-out test metrics, CSV/JSON histories, figures, and the existing centralized comparison on the full training split. Full execution time including that centralized stage was **213.159 s**. Federated-round time was **24.815 s**; HE intervals totalled **14.648 s**.

The default plaintext `main.py --dataset pathmnist --smoke_test --num_clients 3 --rounds 1 --local_epochs 1 --batch_size 32 --lr 0.01` command also completed in the target checkout, including its full centralized baseline, in **191.842 s**. The final complete suite in that checkout passed **48 tests in 24.77 s**.

Because the named repository was outside the initially writable workspace, implementation and the CKKS/paired benchmarks ran first in a writable copy of the local checkout. The tested source and generated artifacts were then applied to the named feature branch, and the complete suite was rerun there.

Checkpoint fields were inspected: only `round`, `model_state`, and `config` are present, all 241,993 floating values are finite, and no private key is serialized by the normal code path. Public contexts and ciphertexts alone use temporary serialization files.

## Numerical precision on identical client states

The offline benchmark aggregates exactly the same randomized synthetic model states with the unchanged plaintext function and the CKKS strategy. For the three-client full-CNN experiment the sample counts are **20, 10, 30**. The two repeats for each of 2, 3 and 5 clients all met `atol=1e-6, rtol=1e-6`:

- Full float32 CNN maximum absolute difference: **2.98023223877e-08**.
- Across those repeats, RMSE ranged from **3.38e-09** to **3.86e-09**.
- A separate mixed-dtype, 8208-value experiment measured float64 tensor maximum absolute differences of **2.21e-10** and **2.38e-10** in two repeats. Float32 differences were **7.45e-09** and **5.59e-09**.
- The maximum relative error with a `1e-6` denominator floor over full-CNN repeats was **0.00420818**. Relative errors are sensitive to cancellation near zero; maximum absolute error and a combined absolute/relative tolerance remain necessary.

This difference includes normal float32 baseline summation/weighting rounding as well as CKKS approximation. It is not a universal error bound. Re-evaluate tolerances after changing values, scale, parameters, dtype or participation. The regular encrypted FL path never retains client plaintext references for error calculation.

## Paired PathMNIST smoke experiments

Both distributions used 3 clients, 2 rounds, 1 local epoch, SGD with learning rate 0.01 and momentum 0.9, batch size 32, seed 42, CPU, and exactly the same first 64 training examples per paired comparison. Partition hashes and identical selected-client schedules are saved in pair manifests. Both use full official validation/test splits.

| Distribution | Plaintext final validation accuracy | CKKS final validation accuracy | Plaintext final validation loss | CKKS final validation loss |
| --- | --- | --- | --- | --- |
| IID | 10.475810% | 10.475810% | 2.1954583733 | 2.1954583657 |
| Non-IID | 10.515794% | 10.515794% | 2.1954646796 | 2.1954646880 |

Final held-out test accuracy also matched within each pair: IID **18.774373%**, non-IID **18.816156%**. Small differences in loss/AUROC remain measurable. Equal classification accuracies do not imply numerically identical model states. Low accuracies are expected from this tiny workload. These two-round curves validate execution; they do not establish converged model quality.

IID average round wall time was **10.579 s** for plaintext and **24.809 s** for CKKS. Average measured CKKS intervals were:

| Operation | Seconds per round |
| --- | --- |
| Packaging | 0.011197 |
| Encoding and encryption | 1.609545 |
| Ciphertext addition and validation | 0.010920 |
| Aggregate recovery and reconstruction | 0.174655 |
| Serialization/deserialization and temporary I/O | 12.332518 |
| Total disjoint measured HE intervals | 14.138835 |

In IID paired training, average serialized ciphertext upload was **59,649,254 bytes** versus **2,903,916 bytes** of estimated raw native floating storage (**20.54×**). The average serialized aggregate was **19,881,158 bytes**. The public context in that experiment was **464,791 bytes**, accounted separately and reused. These are simulated serialized payload sizes, not network measurements; metadata and downloads are excluded from this upload comparison.

Client-count scaling on the full synthetic CNN (two repeats per count):

| Participating clients | Mean HE intervals (seconds) | Mean serialized upload bytes |
| --- | --- | --- |
| 2 | 9.777 | 39,767,094 |
| 3 | 12.926 | 59,643,386 |
| 5 | 20.692 | 99,413,192 |

Temporary-file serialization dominates this implementation on the tested machine. The backend uses the low-level bindings' documented file-based save/load API to avoid automatic unused evaluation-key generation. A future in-memory serialization binding could reduce this cost; this has not been implemented or benchmarked. Memory usage was not profiled.

## Files changed and key responsibilities

Modified existing files:

- `federated_learning/config.py`: optional CKKS settings, structural checks and correct CLI/YAML precedence.
- `federated_learning/federated/client.py`: optional client-side packaging; plaintext local training is retained.
- `federated_learning/federated/server.py`: strategy dispatch in the one existing loop; crypto metrics and effective config output.
- `main.py`: full execution wall-time summary; existing evaluation/centralized flow retained.
- `README.md`: install/run/documentation entry points.
- `.gitignore`: generated local environment, cache and validation-output exclusions. Existing tracked caches remain tracked; test-generated changes to them are restored.

New files:

- `federated_learning/federated/strategies.py`: `PlaintextStrategy`, `CKKSStrategy`, `create_aggregation_strategy`.
- `federated_learning/encryption/interface.py`: typed client/aggregate payloads, timing records and replaceable client/recovery protocols.
- `federated_learning/encryption/ckks_context.py`: `PublicContextBundle`, `PublicContext`, one-time context/key setup, TC128 validation and public/ciphertext serialization.
- `federated_learning/encryption/ckks_encoder.py`: `CKKSEncoder`, deterministic schema/packing/reconstruction and input checks.
- `federated_learning/encryption/ckks_aggregation.py`: `CKKSAggregator`, public-only native ciphertext summation.
- `federated_learning/encryption/trusted_decryptor.py`: `TrustedDecryptor`, authorized aggregate recovery; future collaborative-service replacement point.
- `federated_learning/encryption/__init__.py`: optional package boundary.
- `configs/pathmnist_iid_plaintext.yaml`, `configs/pathmnist_iid_ckks.yaml`, `configs/pathmnist_non_iid_plaintext.yaml`, `configs/pathmnist_non_iid_ckks.yaml`: matched examples.
- `scripts/compare_he.py`: paired existing-loop experiments and plots.
- `scripts/benchmark_ckks.py`: offline same-state precision/client-count benchmarks.
- `tests/test_config_he.py`, `tests/test_ckks.py`, `pytest.ini`: configuration, boundary, regression and optional native HE tests.
- `requirements-he.txt`: optional verified dependency.
- `docs/ckks_fedavg.md`, this report: operation guide and actual results.

## Install and reproduce

Run from the target repository. TenSEAL 0.3.18 has also been installed in the current user's Python environment; plaintext users can omit the HE requirements in a separate environment.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-he.txt
python main.py --dataset pathmnist --aggregation plaintext --smoke_test --num_clients 3 --rounds 1 --local_epochs 1 --batch_size 32 --lr 0.01
python main.py --dataset pathmnist --aggregation ckks --smoke_test --num_clients 3 --rounds 1 --local_epochs 1 --batch_size 32 --lr 0.01
python main.py --config configs/pathmnist_iid_ckks.yaml
python -m pytest tests -q
python -m pytest tests -q -m he
python -m scripts.compare_he --config configs/pathmnist_iid_ckks.yaml --smoke_test --rounds 2 --device cpu --output results/new_paired_smoke
python -m scripts.benchmark_ckks --full_model --client_counts 2 3 5 --repeats 2
```

In the restricted execution environment, writable `--basetemp` and `TEMP`/`TMP` directories were selected. That is an environment workaround, not a different training API. The paired harness deliberately omits the separate centralized baseline; `main.py` retains it. Full five-round configurations are supplied, but long full-dataset encrypted training has not been completed in this task.

## Security assumptions and remaining work

The public aggregation role cannot decrypt and normal execution decrypts only the aggregate. One-process logical roles do not prevent access by a malicious operator with process memory access. Coordinator authorization checks roster/round/routing mistakes but is not cryptographic proof of honest aggregation. No threshold-decryption claim is made. Released aggregates can leak data, and one-client participation reveals that client's recovered update. Poisoning, collusion with the decryptor, malicious server manipulation and network authentication require additional mechanisms.

The pinned TenSEAL build references SEAL 4.3.3, while [Microsoft recommends SEAL 4.4.0 or later for important untrusted-input fixes](https://github.com/microsoft/SEAL). Upgrade and retest before accepting adversarial serialized inputs. This task validates controlled local research only.

Remaining work: isolate roles across authenticated services, update/verify SEAL build provenance, evaluate in-memory serialization, profile memory, test CUDA/university/Colab execution, run longer matched full-dataset experiments, and investigate real collaborative decryption through the existing recovery interface. None of these unmeasured improvements is claimed as completed.

Raw outputs and generated figures are under `results/ckks_validation/` (ignored by Git): `main_ckks/experiment_001`, `paired_iid`, `paired_non_iid`, `precision_full_cnn`, `precision_by_dtype`, plus the default plaintext regression run. They contain plaintext global model checkpoints and metrics, but no secret keys.
