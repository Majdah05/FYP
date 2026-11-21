# Zeroth-order federated training

Training has three independent settings:

| Setting | Values | Default |
| --- | --- | --- |
| `gradient_method` | `backprop`, `zo` | `backprop` |
| `optimizer` | `sgd`, `adam` | `sgd` |
| `aggregation` | `plaintext`, `ckks` | `plaintext` |

Backpropagation, SGD/Adam behavior, and the CKKS aggregation implementation are
preserved. Existing commands and YAML configurations default to backpropagation.
ZO estimates gradients using forward evaluations; SGD or Adam applies those
estimates. ZO-SGD with CKKS therefore uses all three mechanisms together.

## Commands

Run from the repository root after installing `requirements.txt`. Install
`requirements-he.txt` as well when selecting CKKS.

```bash
# Original SGD modes
python main.py --optimizer sgd --aggregation plaintext
python main.py --optimizer sgd --aggregation ckks

# Forward-only local training with SGD updates
python main.py --gradient_method zo --optimizer sgd --aggregation plaintext --lr 0.001
python main.py --gradient_method zo --optimizer sgd --aggregation ckks --lr 0.001

# Average four ZO directions on each minibatch
python main.py --config configs/pathmnist_iid_zo.yaml --zo_directions 4

# Same configuration with encrypted aggregation
python main.py --config configs/pathmnist_iid_zo.yaml --aggregation ckks

# Non-IID data or Adam updates can be selected independently
python main.py --config configs/pathmnist_iid_zo.yaml --distribution non_iid --dirichlet_alpha 0.5
python main.py --config configs/pathmnist_iid_zo.yaml --optimizer adam
```

CLI values override YAML settings only when explicitly supplied. To return a ZO
configuration to backpropagation, add `--gradient_method backprop`.

`--smoke_test --num_clients 3 --rounds 1 --local_epochs 1` limits federated
training to the existing 64-sample subset. As before, `main.py` subsequently
evaluates on the full validation/test loaders and runs its centralized baseline
on the full training set; it is not a fully bounded dataset smoke test.

## Local estimator

`federated_learning/training/zeroth_order.py` implements two-sided simultaneous
perturbation stochastic approximation (SPSA). For each minibatch and each of
`q = zo_directions` directions, sample independent signs `z_i` in `{-1, +1}`
for the trainable parameters and estimate

```text
g_hat = (1/q) sum_i [(L(theta + mu*z_i) - L(theta - mu*z_i)) / (2*mu)] * z_i
```

The selected PyTorch optimizer consumes `g_hat` in `parameter.grad`. SGD retains
the configured momentum, and Adam retains its normal moment updates. ZO never
calls `backward()` and never builds a backward graph.

- `zo_mu` is the finite, positive perturbation size (default `0.001`).
- `zo_directions` is a positive integer (default `1`). More directions increase
  the evaluation budget and can reduce estimator variance.
- Each minibatch uses `2*q + 1` forward evaluations: `2*q` perturbed loss
  evaluations plus one unperturbed evaluation for training-loss logging and
  normal model-buffer updates. All use the same minibatch.
- Perturbed evaluations share dropout randomness, and each starts with the same
  model buffers. Parameters and buffers are restored exactly after probing,
  including when a probe raises an exception. Only the unperturbed evaluation
  advances the global model RNG and updates buffers such as BatchNorm statistics.
- Frozen parameters are not perturbed and receive no estimated gradients.
- Perturbations use a dedicated seeded generator. Federated seeds are derived
  from the experiment seed, round index, and client ID. The centralized baseline
  uses the experiment seed for its own perturbation sequence.

The centralized baseline also uses the configured gradient method, perturbation
settings, and optimizer. Its existing epoch-budget calculation remains unchanged.
Configuration files and checkpoints record the ZO settings, while federated and
centralized histories identify `gradient_method` and `optimizer`.

## Aggregation and experiments

Clients still send complete updated model states in plaintext mode. With CKKS,
clients encrypt their sample-weighted model contributions using the existing
encoder; the server sums ciphertexts and the trusted decryptor recovers the
aggregate. ZO does not change that protocol or its security assumptions.

ZO does not itself encrypt updates or reduce the number of model values sent.
Its convergence, time, and memory use must be measured; optimizer state,
parameter snapshots, directions, and estimated gradients still use memory even
though no backward graph is built. Full CNN training from random initialization
may require more directions and different learning rates than backpropagation.
No PathMNIST accuracy or speed improvement is implied by these starting settings.

Compare backpropagation/plaintext with backpropagation/CKKS to isolate encryption
overhead, and backpropagation/plaintext with ZO/plaintext to study optimization.
Include ZO/CKKS to measure their combined behavior. Match data partitions,
participation, and seeds, and report forward-evaluation budgets as well as rounds
and wall time. Tune on validation data and retain the test split for final reports.

The ZO-SGD approach is also discussed in
[Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333).
This implementation uses a Rademacher SPSA estimator, not the paper's full MeZO
fine-tuning implementation or its benchmark settings.

## Verification

```bash
# Default suite, including ZO plaintext tests
python -m pytest tests -q

# All tests, including actual CKKS training and ZO/CKKS comparisons
python -m pytest tests -q -o addopts=
```

Tests cover scalar gradient correctness, dropout randomness, buffer updates,
restoration after failed probes, frozen parameters, deterministic runs,
forward-only client and centralized training, IID/non-IID partial participation,
all four SGD modes, encrypted-only client uploads, and complete `main.py` ZO
runs on synthetic data. CKKS tests require the optional HE dependency. These
checks verify functionality rather than PathMNIST convergence or GPU performance.
