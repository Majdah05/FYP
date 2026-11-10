# Plaintext FedAvg vs CKKS-encrypted FedAvg on PathMNIST

These experiments test what CKKS encryption costs. It is Microsoft SEAL via TenSEAL 0.3.18, with a **third-party decryptor**, compared against the plaintext baseline. The comparison looks at **accuracy change, added time / CPU cost, and bandwidth**.

## Design

```
client k :  c_k = Enc_pk(n_k · w_k)              (public key only)
server   :  C   = Σ_k c_k                       (ciphertext addition, public key only)
decryptor:  w   = Dec_sk(C) / Σ_k n_k           (only party holding the secret key; sees only the sum)
```

The code is the repo's own `CKKSStrategy`, `CKKSAggregator` and `TrustedDecryptor`, used unchanged. `run_fl.py` only adds measurement hooks:

- per-round CPU and wall time
- a plaintext *shadow* FedAvg on the same client models, which measures the real CKKS error (it is never used for training)
- peak RAM

## Experiments

The settings match the best baseline pair: **T09 (IID) / T10 (non-IID)**.

| Run | Distribution | Aggregation |
|---|---|---|
| 1 | IID | plaintext |
| 2 | IID | CKKS |
| 3 | non-IID (Dirichlet α = 0.5) | plaintext |
| 4 | non-IID (Dirichlet α = 0.5) | CKKS |

All four runs use 5 clients, 20 rounds, 5 local epochs, batch size 128, SGD with lr 0.01 and momentum 0.9, and seed 42.

Each plaintext/CKKS twin shares the seed, partition, initial model and client order. Kernels are deterministic. The only difference between twins is the aggregation.

The HE cost does not depend on lr, epochs or batch size. It depends on model size and client count, which `he_cost_bench.py` covers:

- clients: 2 / 3 / 5 / 10
- CKKS parameter variants
- RAM-disk serialization

## How to run

**Option A: Colab (recommended, GPU).** Open `PathMNIST_CKKS_vs_Plaintext.ipynb`, select a T4 GPU and choose Run all. It takes about 1h45. The notebook is self-contained.

**Option B: locally, from the repo root.**

```powershell
python -m pip install -r requirements.txt -r requirements-he.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\ckks_vs_plaintext\run_experiments.ps1 -Device auto
```

- PathMNIST is downloaded to `data/` if it is missing.
- Finished runs are skipped on re-run.
- On CPU, each run takes about 1–2 h.

Single run:

```bash
python -I experiments/ckks_vs_plaintext/run_fl.py --repo . --data_dir data --download \
  --out experiments/ckks_vs_plaintext/results/runs/iid_ckks --mode ckks --distribution iid --device auto
```

## Outputs (`results/`)

| File | Content |
|---|---|
| `runs/<dist>_<mode>/federated_history.json/.csv` | per-round metrics, HE timings, ciphertext bytes, CKKS-vs-plaintext error, test metrics |
| `summary.json` | paired deltas: Δ accuracy/F1/AUROC/loss, added wall and CPU time, upload and total-traffic expansion, extra RAM |
| `ckks_vs_plaintext_report.html` | self-contained report with charts; GPU T09/T10 baseline shown as reference rows |
| `he_cost_benchmark/*.json` | HE-only measurements (already measured; see below) |

## HE cost already measured

These figures cover one aggregation round on the real 241,993-parameter CNN. They were measured on a 2-core Linux CPU with N = 8192, q = [60,40,40,60] and Δ = 2^40, averaged over 3 repeats.

| Clients | CKKS round | Ciphertext upload | Plaintext upload | Max error vs plaintext |
|---|---|---|---|---|
| 2 | 1.40 s | 39.8 MB | 1.9 MB | 3.0e-8 |
| 3 | 1.92 s | 59.6 MB | 2.9 MB | 2.5e-8 |
| 5 | 3.01 s | 99.4 MB | 4.8 MB | 3.0e-8 |
| 10 | 5.36 s | 198.8 MB | 9.7 MB | 3.0e-8 |

- **Per-client upload.** Each client sends 60 ciphertexts, about 19.9 MB per round. The float32 model is 0.97 MB, so the upload grows **20.5×**.
- **Total traffic per round (5 clients).** It is about 125 MB with CKKS against 9.7 MB in plaintext, which is **12.9×**.
- **Plaintext FedAvg time.** Plaintext aggregation takes about 2.5 ms.
- **Parameters.** Aggregation only adds ciphertexts and never multiplies them. With q = [60,40,60], the upload falls to **14.1 MB per client (−29%)** and the round takes about 28% less time. The error stays the same.
- **Serialization.** The earlier Windows measurement spent 12.3 s of its 14.1 s in serialization. That time went into temporary-file I/O for SEAL's path-based save/load, not into CKKS. On Linux, serialization takes about 1.2 s per 5-client round.
