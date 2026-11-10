"""Summarise paired plaintext vs CKKS runs + HE benchmarks into one JSON."""
import json, sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])          # dir containing iid_plaintext, iid_ckks, non_iid_plaintext, non_iid_ckks
bench = Path(sys.argv[2])         # dir with he_*.json
out = Path(sys.argv[3])

PT_MODEL_BYTES = 241993 * 4


def load(name):
    p = root / name / 'federated_history.json'
    return json.loads(p.read_text()) if p.exists() else None


summary = {'runs': {}, 'pairs': {}}
for dist in ('iid', 'non_iid'):
    pair = {}
    for mode in ('plaintext', 'ckks'):
        d = load(f'{dist}_{mode}')
        if d is None:
            continue
        h, t = d['history'], d['test_metrics']
        k = len(h[0]['selected_client_ids'])
        r = {
            'rounds_completed': len(h),
            'test_accuracy': t['test_accuracy'], 'test_macro_f1': t['macro_f1'], 'test_auroc': t['auroc'],
            'test_loss': t['test_loss'], 'test_precision': t['precision'], 'test_recall': t['recall'],
            'val_acc_curve': [x['validation_accuracy'] for x in h],
            'val_loss_curve': [x['validation_loss'] for x in h],
            'val_f1_curve': [x['macro_f1'] for x in h],
            'train_loss_curve': [x['training_loss'] for x in h],
            'round_wall_curve': [x['round_time_seconds'] for x in h],
            'round_cpu_curve': [x.get('round_cpu_seconds') for x in h],
            'mean_round_wall_s': float(np.mean([x['round_time_seconds'] for x in h])),
            'mean_round_cpu_s': float(np.mean([x['round_cpu_seconds'] for x in h])),
            'federated_wall_s': d['federated_wall_seconds'], 'federated_cpu_s': d['federated_cpu_seconds'],
            'peak_rss_mb': d['peak_rss_mb'],
            'mean_aggregation_call_wall_s': float(np.mean([x['aggregation_call_wall_seconds'] for x in h])),
            'clients': k,
            # bandwidth per round
            'upload_bytes_per_round': (float(np.mean([x['he_ciphertext_upload_bytes'] for x in h])) if mode == 'ckks'
                                       else k * PT_MODEL_BYTES),
            'download_bytes_per_round': k * PT_MODEL_BYTES,   # global model broadcast is plaintext in both designs
            'server_to_decryptor_bytes_per_round': (float(np.mean([x['he_aggregate_ciphertext_bytes'] for x in h])) if mode == 'ckks' else 0),
            'decryptor_to_server_bytes_per_round': PT_MODEL_BYTES if mode == 'ckks' else 0,
            'one_time_public_context_bytes_per_client': h[0].get('he_public_context_bytes', 0),
        }
        r['total_bytes_per_round'] = (r['upload_bytes_per_round'] + r['download_bytes_per_round'] +
                                      r['server_to_decryptor_bytes_per_round'] + r['decryptor_to_server_bytes_per_round'])
        if mode == 'ckks':
            for key in ('he_packaging_time_seconds', 'he_encryption_time_seconds', 'he_aggregation_time_seconds',
                        'he_decryption_time_seconds', 'he_serialization_time_seconds', 'he_total_time_seconds',
                        'client_encrypt_cpu_seconds', 'aggregation_call_cpu_seconds'):
                r['mean_' + key] = float(np.mean([x[key] for x in h]))
            r['he_ciphertext_count_per_round'] = h[0]['he_ciphertext_count']
            r['per_round_max_abs_err'] = [x['ckks_vs_plaintext_max_abs_err'] for x in h]
            r['per_round_rmse'] = [x['ckks_vs_plaintext_rmse'] for x in h]
            r['max_abs_weight'] = max(x['max_abs_weight'] for x in h)
        summary['runs'][f'{dist}_{mode}'] = r
        pair[mode] = r
    if len(pair) == 2:
        p, c = pair['plaintext'], pair['ckks']
        summary['pairs'][dist] = {
            'delta_test_accuracy_pp': 100 * (c['test_accuracy'] - p['test_accuracy']),
            'delta_test_macro_f1_pp': 100 * (c['test_macro_f1'] - p['test_macro_f1']),
            'delta_test_auroc': c['test_auroc'] - p['test_auroc'],
            'delta_test_loss': c['test_loss'] - p['test_loss'],
            'max_abs_val_acc_diff_pp_any_round': 100 * max(abs(a - b) for a, b in zip(c['val_acc_curve'], p['val_acc_curve'])),
            'added_wall_s_per_round': c['mean_round_wall_s'] - p['mean_round_wall_s'],
            'wall_overhead_pct': 100 * (c['federated_wall_s'] / p['federated_wall_s'] - 1),
            'added_cpu_s_per_round': c['mean_round_cpu_s'] - p['mean_round_cpu_s'],
            'cpu_overhead_pct': 100 * (c['federated_cpu_s'] / p['federated_cpu_s'] - 1),
            'upload_expansion_x': c['upload_bytes_per_round'] / p['upload_bytes_per_round'],
            'total_traffic_expansion_x': c['total_bytes_per_round'] / p['total_bytes_per_round'],
            'extra_peak_rss_mb': c['peak_rss_mb'] - p['peak_rss_mb'],
        }

bench_rows = []
for f in sorted(bench.glob('he_*.json')):
    if f.name == 'he_smoke.json':
        continue
    bench_rows += json.loads(f.read_text())
agg = {}
for r in bench_rows:
    key = f"{r['label']}|{r['params']}|{r['clients']}"
    agg.setdefault(key, []).append(r)
summary['bench'] = {}
for key, rows in agg.items():
    label, params, clients = key.split('|')
    summary['bench'][key] = {'label': label, 'params': params, 'clients': int(clients), 'n': len(rows),
                             **{m: float(np.mean([r[m] for r in rows])) for m in rows[0] if isinstance(rows[0][m], (int, float)) and m not in ('clients', 'repeat')},
                             'ckks_round_wall_s_std': float(np.std([r['ckks_round_wall_s'] for r in rows]))}
out.write_text(json.dumps(summary, indent=1))
print(json.dumps(summary['pairs'], indent=1))
