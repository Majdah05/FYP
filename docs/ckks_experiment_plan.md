# PathMNIST federated learning experiments with CKKS

Run the existing PathMNIST CNN with federated learning and CKKS-encrypted
aggregation. Each client trains locally with PyTorch, encrypts its sample-weighted
model weights, and sends ciphertexts for homomorphic aggregation. The trusted
decryptor recovers the aggregated global model.

## The six experiments

The [batch script](../run_ckks_experiments.ps1) executes these runs sequentially:

| Run | Aggregation | Data distribution | Random seed |
| --- | --- | --- | --- |
| 1 | CKKS | IID | 42 |
| 2 | CKKS | IID | 43 |
| 3 | CKKS | IID | 44 |
| 4 | CKKS | Non-IID | 42 |
| 5 | CKKS | Non-IID | 43 |
| 6 | CKKS | Non-IID | 44 |

All six use the full PathMNIST training split. Validation and final testing use
the official held-out splits. The runner explicitly disables smoke-test mode.

## Default settings

| Setting | Value |
| --- | --- |
| Model | Existing PathMNIST CNN |
| Clients | 3 |
| Participation | All clients in every round |
| Communication rounds | 20 |
| Local epochs per round | 1 |
| Batch size | 32 |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Non-IID Dirichlet alpha | 0.5 |
| Training device | CUDA when available, otherwise CPU |
| CKKS device | CPU |
| Polynomial modulus degree | 8192 |
| Coefficient modulus bit sizes | [60, 40, 40, 60] |
| Encoding scale | 2**40 |
| Chunk size | 4096 |

## Why repeat with different seeds?

A training seed controls reproducible random choices, including the CNN's
starting weights, the allocation of training examples to clients, and dropout.
Different seeds can therefore produce different results with the same training
settings.

Seeds 42, 43 and 44 provide three repetitions for each distribution. Report the
mean and standard deviation of the results across these repetitions to show
variability. These particular seed numbers are arbitrary; they are not tuned
hyperparameters or CKKS encryption keys. SEAL key generation uses its own
cryptographic randomness.

Using seed 42 alone is useful for an initial check. The batch script retains all
three seeds for the six-run evaluation.

## Fair comparison with plaintext federated learning

For every encrypted experiment, compare against a plaintext experiment with the
same seed, distribution, model, data preprocessing, client count, participation,
round count, local epochs, batch size, optimizer, learning rate and momentum.
Keep the Dirichlet alpha identical for the non-IID comparisons.

For example, compare CKKS IID seed 42 against plaintext IID seed 42. Existing
plaintext results can be reused when their settings match. The six-run script
runs encrypted experiments; matching plaintext experiments must be available
separately for the comparison.

## Install and run

Open PowerShell in the repository root, where `main.py` is located:

```powershell
python -m pip install -r requirements.txt -r requirements-he.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_ckks_experiments.ps1
```

Preview the six commands without starting training:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_ckks_experiments.ps1 -DryRun
```

Check dependencies, configurations and SEAL security parameters:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_ckks_experiments.ps1 -CheckOnly
```

Override settings to match a different plaintext baseline, for example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_ckks_experiments.ps1 -NumClients 5 -LocalEpochs 3 -BatchSize 64 -LearningRate 0.01 -Device cuda
```

The script chooses `.venv\Scripts\python.exe` when it exists, otherwise `python`
on PATH. Supply `-PythonPath 'C:\path\to\python.exe'` to select an interpreter.
Copy the repository to another machine and run the same command from its root;
the runner resolves the project folder from its own location.

## Outputs and execution behavior

Each invocation creates a new timestamped batch under `results/ckks_six_runs/`.
It saves the run plan, environment information, per-experiment logs and
`summary.csv`, alongside each experiment's metrics, model checkpoints and plots.
Recorded metrics include validation accuracy/loss, macro F1, held-out test
performance, round time, encryption/decryption/serialization time and serialized
ciphertext payload sizes.

The existing `main.py` also completes its centralized baseline after each
federated run. Allow time for that additional stage. The script records failed
experiments and continues with the remaining runs. Rerunning creates a new batch;
it does not resume interrupted training.

For CKKS architecture, metric definitions and security limitations, see
[CKKS FedAvg](ckks_fedavg.md). This is a single-process research implementation
with a trusted decryptor.
