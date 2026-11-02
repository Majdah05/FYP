# Federated Learning Baseline with FedAvg

This project implements a clean, configurable federated learning baseline for PathMNIST and an IDS dataset interface. The initial focus is a reproducible Federated Averaging (FedAvg) simulation using PyTorch, with centralized training as a reference point.

## 1. Installation

```bash
python -m pip install -r requirements.txt
```

Use the project root as the working directory.

## 2. PathMNIST preparation

The implementation uses the official MedMNIST package. If the dataset is not already downloaded, the code downloads it automatically when you run the training script.

```bash
python main.py --dataset pathmnist --smoke_test --num_clients 3 --rounds 1 --local_epochs 1 --batch_size 32 --lr 0.01
```

The full centralized baseline and federated runs can then be executed normally.

## 3. Centralized baseline

```bash
python main.py --dataset pathmnist --num_clients 5 --rounds 5 --local_epochs 2 --batch_size 64 --lr 0.01 --distribution iid --model cnn
```

This records training loss, validation loss, validation accuracy, macro F1, and other metrics for the centralized PathMNIST model.

## 4. Federated training

```bash
python main.py --dataset pathmnist --num_clients 5 --rounds 20 --local_epochs 2 --batch_size 64 --lr 0.01 --distribution iid
```

```bash
python main.py --dataset pathmnist --num_clients 10 --rounds 50 --local_epochs 5 --batch_size 32 --lr 0.001 --distribution non_iid --dirichlet_alpha 0.5
```

The code uses a global model, randomly or Dirichlet-partitioned client data, local client updates, and FedAvg aggregation.

## 5. Configurations

A YAML config can be used as a base and CLI flags override it:

```bash
python main.py --config configs/pathmnist_iid.yaml --rounds 10
```

Examples:

- `configs/pathmnist_iid.yaml`
- `configs/pathmnist_non_iid.yaml`

## 6. FedAvg flow

FedAvg follows the standard communication round:

1. The server chooses participating clients.
2. A copy of the current global model is sent to each client.
3. Each client updates the model on its local data.
4. The client sends its updated parameters and sample count.
5. The server aggregates using sample-size weighting.
6. The global model is replaced by the aggregated model.
7. Validation is computed on the updated global model.

The aggregation formula is:

$$
w_{t+1} = \sum_{k \in S_t} \frac{n_k}{\sum_{j \in S_t} n_j} w_{t+1}^{k}
$$

This is implemented in `federated_learning/federated/aggregation.py` and is mathematically equivalent to weighted averaging over the selected clients.

The smoke test partitions only the first 64 training samples. The centralized baseline still uses the full official training split.

- IID: randomly shuffles and splits the training data across clients.
- Non-IID: uses a Dirichlet distribution for class skew across clients.

Each client receives a disjoint subset, and the total dataset is preserved without duplication.

## 8. Evaluation and outputs

After federated training finishes, the final global model is evaluated once on the untouched PathMNIST test split. Its metrics are saved to `federated_test_metrics.json` and included under `federated_test` in `comparison_summary.json`. Per-round validation metrics remain separate; use validation data for model selection and test metrics for final reporting.

Each experiment produces a unique output folder under `results/` with:
The server runs the configured number of local epochs and uses the selected optimizer. For SGD, `--momentum` defaults to `0.9`; Adam does not use this setting. The centralized comparison uses an epoch budget matched to the configured number of rounds, local epochs, and participation fraction.

- config JSON
- train/validation metrics CSV
- JSON summaries
- validation curves
- client distribution plots
- checkpoints

The `scripts/compare_experiments.py` script scans completed experiment directories and produces a simple comparison plot.

## 9. IDS integration

The exact IDS dataset schema is not supplied in the workspace, so the integration expects:

- dataset path
- target column name
- numeric columns
- categorical columns
- optional preprocessing configuration

This keeps the federated algorithm separate from IDS-specific preprocessing and feature engineering.

## 10. Security architecture

Plaintext aggregation remains the default. Optional Microsoft SEAL CKKS aggregation is now available through TenSEAL. It encrypts sample-weighted client model contributions, sums serialized ciphertexts on a public-only server, and recovers only the authorized aggregate with a trusted decryptor. Local training and evaluation remain normal PyTorch operations. See [CKKS FedAvg](docs/ckks_fedavg.md) for the security assumptions and measured overhead.

The project can later support additional mechanisms such as:

- secure aggregation
- differential privacy
- gradient clipping
- robust aggregation such as median or trimmed mean
- poisoning experiments
- communication encryption

The architecture is prepared for these additions by keeping client local training, server aggregation, and evaluation as separate modules.

````text
federated_learning/
├── main.py
├── README.md
├── configs/
├── datasets/
├── models/
├── federated/
├── training/
├── utils/
├── scripts/
├── tests/

```bash
python -m pytest tests -q
````

These checks cover:

- IID partitioning does not duplicate or lose samples
- non-IID partitioning preserves client assignment integrity

Reproducibility is controlled through the `seed` parameter. For practical experiments, avoid nondeterminism from GPU scheduling or external dataset order issues by fixing seeds and using CPU when comparing results across repeated runs.

## 14. Research interpretation

Use the experiment outputs to compare:

- centralized vs federated learning accuracy
- IID vs non-IID behavior
- local epoch count and participation effects
- communication rounds and speed trade-offs

The project is intentionally modular so that the same pipeline can later be extended with security and privacy mechanisms.

## Optional CKKS aggregation

See the [six-run PathMNIST experiment plan](docs/ckks_experiment_plan.md) for the experiment list, seed explanation and batch-run commands.

```bash
python -m pip install -r requirements-he.txt
python main.py --aggregation ckks --smoke_test --num_clients 3 --rounds 1 --local_epochs 1 --batch_size 32 --lr 0.01
python main.py --config configs/pathmnist_iid_ckks.yaml
```

Omit `--aggregation` or use `--aggregation plaintext` for the original FedAvg function without importing TenSEAL. CLI values explicitly supplied by the user override YAML; missing CLI values preserve YAML settings. `main.py` still runs its full centralized comparison after federated smoke training.

For matched federated runs and offline precision benchmarks:

```bash
python -m scripts.compare_he --config configs/pathmnist_iid_ckks.yaml --smoke_test --rounds 2 --device cpu --output results/paired_smoke
python -m scripts.benchmark_ckks --full_model --client_counts 2 3 5 --repeats 3
python -m pytest tests -q -m he
```

See [installation, architecture and metric definitions](docs/ckks_fedavg.md) and [actual validation results](docs/ckks_validation_report.md). CKKS is approximate; this one-process research simulation does not provide process isolation, threshold decryption, or protection against a malicious coordinator. The verified TenSEAL pin references SEAL 4.3.3, which needs an upgrade and revalidation before exposing deserialization to untrusted inputs.
