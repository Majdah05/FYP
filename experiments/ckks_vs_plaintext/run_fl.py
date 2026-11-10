"""Run one PathMNIST FedAvg experiment (plaintext or CKKS) with the repository's
own training loop, adding measurement-only instrumentation:

* per-round process CPU time and wall time
* plaintext aggregation time (plaintext mode)
* in CKKS mode, a shadow plaintext fedavg_aggregate on the SAME client states,
  giving the real per-round CKKS aggregation error on trained weights
  (measurement only; the shadow result is never used by training)
* peak resident memory

Data: PathMNIST is read from a local pathmnist.npz (download disabled). Images
are normalised once into memory with exactly the repository's ToTensor +
Normalize arithmetic (verified bit-exact at start-up), to avoid per-image PIL
overhead on CPU. Partitioning, client loaders (shuffle=False as in the repo),
model, optimiser and evaluation are the repository's code.
"""
import argparse, copy, json, resource, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['plaintext', 'ckks'], required=True)
    ap.add_argument('--distribution', choices=['iid', 'non_iid'], required=True)
    ap.add_argument('--num_clients', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=20)
    ap.add_argument('--local_epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--train_limit', type=int, default=0, help='debug only')
    ap.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    ap.add_argument('--download', action='store_true', help='let MedMNIST download pathmnist.npz into data_dir')
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo).resolve()))

    import numpy as np, torch
    torch.set_num_threads(a.threads)
    # deterministic kernels so the plaintext/CKKS twins differ only by aggregation numerics
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    import medmnist
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from main import PathMNISTSubsetFactory, set_seed, evaluate_federated_test
    from federated_learning.config import ExperimentConfig
    from federated_learning.federated import server as server_mod
    from federated_learning.federated import strategies as strat_mod
    from federated_learning.federated.aggregation import fedavg_aggregate
    from federated_learning.federated.server import initialize_global_model, run_federated_training
    from federated_learning.utils.logging import save_json, save_metrics_csv

    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    mean = [0.740545, 0.532982, 0.705829]; std = [0.123682, 0.176763, 0.124431]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    class MemDS(torch.utils.data.Dataset):
        def __init__(self, split):
            Path(a.data_dir).mkdir(parents=True, exist_ok=True)
            raw = medmnist.PathMNIST(root=a.data_dir, split=split, transform=tf, download=a.download)
            n = len(raw)
            m = torch.tensor(mean).view(1, 3, 1, 1); s = torch.tensor(std).view(1, 3, 1, 1)
            self.x = torch.empty((n, 3, 28, 28), dtype=torch.float32)
            for start in range(0, n, 8192):   # chunked to bound peak memory
                chunk = torch.from_numpy(raw.imgs[start:start + 8192]).permute(0, 3, 1, 2).float().div(255.0)
                self.x[start:start + 8192] = chunk.sub_(m).div_(s)
            for i in (0, 1, n // 2, n - 1):
                if not torch.equal(raw[i][0], self.x[i]):
                    raise RuntimeError(f'in-memory preprocessing differs from repo transform at {split}[{i}]')
            self.labels = raw.labels.copy()
            self.y = torch.from_numpy(raw.labels.astype(np.int64))
            del raw
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.x[i], self.y[i]

    train, val, test = MemDS('train'), MemDS('val'), MemDS('test')
    val_loader = DataLoader(val, batch_size=a.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=a.batch_size, shuffle=False)
    idx = list(range(a.train_limit)) if a.train_limit else None
    factory = PathMNISTSubsetFactory(train, batch_size=a.batch_size, seed=a.seed, sample_indices=idx)

    cfg = ExperimentConfig(num_clients=a.num_clients, rounds=a.rounds, local_epochs=a.local_epochs,
                           batch_size=a.batch_size, lr=a.lr, momentum=0.9, optimizer='sgd',
                           distribution=a.distribution, dirichlet_alpha=0.5, client_fraction=1.0,
                           model='cnn', seed=a.seed, device=device, output_dir=str(out),
                           save_checkpoints=False, aggregation=a.mode)
    cfg.validate()

    # ---------------- measurement-only instrumentation ----------------
    extra = {}            # round -> dict of extra metrics
    stash = {}            # (round, client) -> plaintext state copy (CKKS shadow only)
    cpu_mark = {'t': None, 'r': None}
    orig_create = strat_mod.create_aggregation_strategy

    def create(config, initial_state):
        st = orig_create(config, initial_state)
        orig_begin, orig_agg = st.begin_round, st.aggregate

        def begin_round(round_id, participants):
            cpu_mark['t'] = time.process_time(); cpu_mark['r'] = round_id
            extra.setdefault(round_id, {})
            return orig_begin(round_id, participants)

        def aggregate(updates, counts, client_metrics):
            rid = cpu_mark['r']
            t, c = time.perf_counter(), time.process_time()
            state, m = orig_agg(updates, counts, client_metrics)
            e = extra[rid]
            e['aggregation_call_wall_seconds'] = time.perf_counter() - t   # server + TTP (CKKS) / fedavg (plain)
            e['aggregation_call_cpu_seconds'] = time.process_time() - c
            if config.aggregation == 'ckks':
                shadow = fedavg_aggregate([stash.pop((rid, u.client_id)) for u in updates], counts)
                ref = np.concatenate([v.double().numpy().ravel() for v in shadow.values() if v.is_floating_point()])
                got = np.concatenate([v.double().numpy().ravel() for v in state.values() if v.is_floating_point()])
                err = np.abs(ref - got)
                e['ckks_vs_plaintext_max_abs_err'] = float(err.max())
                e['ckks_vs_plaintext_rmse'] = float(np.sqrt((err ** 2).mean()))
                e['max_abs_weight'] = float(np.abs(ref).max())
            return state, m

        st.begin_round, st.aggregate = begin_round, aggregate
        if st.client_encoder is not None:
            enc = st.client_encoder
            orig_prepare = enc.prepare

            def prepare(state, sample_count, client_id, round_id):
                stash[(round_id, client_id)] = {k: v.detach().clone() for k, v in state.items()}
                c = time.process_time()
                r = orig_prepare(state, sample_count, client_id, round_id)
                extra[round_id]['client_encrypt_cpu_seconds'] = extra[round_id].get('client_encrypt_cpu_seconds', 0.0) + time.process_time() - c
                return r
            enc.prepare = prepare
        return st

    server_mod.create_aggregation_strategy = create
    orig_eval = server_mod.evaluate_model

    def eval_hook(model, loader, device='cpu', criterion=None):
        # called once per round right after aggregation: close the round's CPU window
        res = orig_eval(model, loader, device=device, criterion=criterion)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if cpu_mark['t'] is not None:
            extra[cpu_mark['r']]['round_cpu_seconds'] = time.process_time() - cpu_mark['t']
        return res
    server_mod.evaluate_model = eval_hook
    # ------------------------------------------------------------------

    set_seed(a.seed)
    model = initialize_global_model('pathmnist', model_name='cnn', num_classes=9)
    t0, c0 = time.perf_counter(), time.process_time()
    model, history = run_federated_training(factory, val_loader, cfg, output_dir=str(out), global_model=model)
    fl_wall, fl_cpu = time.perf_counter() - t0, time.process_time() - c0
    for h in history:
        h.update(extra.get(h['round'], {}))
    test_metrics = evaluate_federated_test(model, test_loader, cfg, out)
    save_json({'config': cfg.__dict__, 'history': history, 'test_metrics': test_metrics,
               'federated_wall_seconds': fl_wall, 'federated_cpu_seconds': fl_cpu,
               'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               'torch_threads': torch.get_num_threads(), 'device': device,
               'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None},
              out / 'federated_history.json')
    save_metrics_csv(history, out / 'federated_history.csv')
    torch.save(model.state_dict(), out / 'final_model.pt')
    print('DONE', a.mode, a.distribution, 'test_acc', test_metrics['test_accuracy'], 'wall', fl_wall, flush=True)


if __name__ == '__main__':
    main()
