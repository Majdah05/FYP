"""HE-only cost benchmark on the real PathMNIST CNN schema (no dataset needed).

Uses the repository's own CKKSStrategy (client encoder -> public aggregator ->
trusted third-party decryptor) exactly as training does, and measures for one
FedAvg round: wall time, CPU time, per-phase timings, ciphertext bytes and
aggregation error against the plaintext fedavg_aggregate on the same states.

Usage (from repo root):
  python -I /path/he_cost_bench.py --repo . --out OUT.json --clients 2 3 5 10 \
      --params "8192:60,40,40,60:40" --repeats 3
"""
import argparse, copy, json, os, resource, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--clients', type=int, nargs='+', default=[5])
    ap.add_argument('--params', nargs='+', default=['8192:60,40,40,60:40'])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--samples_per_client', type=int, default=17999)
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo).resolve()))
    import numpy as np, torch
    torch.set_num_threads(1)
    from federated_learning.config import ExperimentConfig
    from federated_learning.federated.aggregation import fedavg_aggregate
    from federated_learning.federated.strategies import create_aggregation_strategy
    from federated_learning.models.cnn import PathMNISTCNN

    torch.manual_seed(0)
    template = PathMNISTCNN(9).state_dict()
    rng = np.random.default_rng(0)
    records = []
    rid = 0
    for p in a.params:
        n, bits, scale = p.split(':')
        cfg = ExperimentConfig(aggregation='ckks', ckks_poly_modulus_degree=int(n),
                               ckks_coeff_mod_bit_sizes=[int(b) for b in bits.split(',')],
                               ckks_global_scale_power=int(scale))
        t0 = time.perf_counter()
        strat = create_aggregation_strategy(cfg, template)
        init_s = time.perf_counter() - t0
        for k in a.clients:
            for rep in range(a.repeats):
                rid += 1
                states = []
                for _ in range(k):
                    s = copy.deepcopy(template)
                    for name, v in s.items():
                        if v.is_floating_point():
                            # realistic magnitudes: trained CNN weights are mostly |w|<0.5
                            s[name] = (v + torch.from_numpy(rng.normal(0, 0.05, tuple(v.shape))).float())
                    states.append(s)
                counts = [a.samples_per_client + i for i in range(k)]
                t = time.perf_counter(); c = time.process_time()
                plain = fedavg_aggregate(states, counts)
                plain_wall, plain_cpu = time.perf_counter() - t, time.process_time() - c
                strat.begin_round(rid, tuple(enumerate(counts)))
                t = time.perf_counter(); c = time.process_time()
                enc_wall_clients, enc_cpu_clients, packs = [], [], []
                for i, (s, cnt) in enumerate(zip(states, counts)):
                    t1 = time.perf_counter(); c1 = time.process_time()
                    packs.append(strat.client_encoder.prepare(s, cnt, i, rid))
                    enc_wall_clients.append(time.perf_counter() - t1); enc_cpu_clients.append(time.process_time() - c1)
                t2 = time.perf_counter(); c2 = time.process_time()
                agg_state, m = strat.aggregate([x[0] for x in packs], counts, [x[1] for x in packs])
                server_ttp_wall, server_ttp_cpu = time.perf_counter() - t2, time.process_time() - c2
                wall, cpu = time.perf_counter() - t, time.process_time() - c
                ref = np.concatenate([v.double().numpy().ravel() for v in plain.values() if v.is_floating_point()])
                got = np.concatenate([v.double().numpy().ravel() for v in agg_state.values() if v.is_floating_point()])
                err = np.abs(ref - got)
                rec = dict(label=a.label, params=p, clients=k, repeat=rep + 1,
                           tmpdir=os.environ.get('TMPDIR', '/tmp(default)'),
                           context_init_seconds=init_s,
                           plaintext_agg_wall_s=plain_wall, plaintext_agg_cpu_s=plain_cpu,
                           ckks_round_wall_s=wall, ckks_round_cpu_s=cpu,
                           per_client_encrypt_wall_s=float(np.mean(enc_wall_clients)),
                           per_client_encrypt_cpu_s=float(np.mean(enc_cpu_clients)),
                           server_plus_ttp_wall_s=server_ttp_wall, server_plus_ttp_cpu_s=server_ttp_cpu,
                           max_abs_err=float(err.max()), rmse=float(np.sqrt((err ** 2).mean())),
                           plaintext_upload_bytes=k * sum(v.numel() * v.element_size() for v in template.values() if v.is_floating_point()),
                           peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                           **{kk: vv for kk, vv in m.items() if not isinstance(vv, str)})
                records.append(rec)
                print(json.dumps({x: rec[x] for x in ('params', 'clients', 'repeat', 'ckks_round_wall_s', 'he_serialization_time_seconds',
                                                      'he_ciphertext_upload_bytes', 'max_abs_err')}), flush=True)
    Path(a.out).write_text(json.dumps(records, indent=1))


if __name__ == '__main__':
    main()
