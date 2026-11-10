"""Build a self-contained offline HTML report from analyze.py's summary.json."""
import html, json, math, sys
from pathlib import Path

S = json.loads(Path(sys.argv[1]).read_text())
OUT = Path(sys.argv[2])
runs, pairs, bench = S['runs'], S['pairs'], S.get('bench', {})

# GPU plaintext baseline from the 26 Oct 2026 report (T09 IID / T10 non-IID)
BASE = {'iid': dict(name='T09 (GPU baseline)', acc=.865042, f1=.824045, auroc=.979282, loss=.943519, minutes=22.4),
        'non_iid': dict(name='T10 (GPU baseline)', acc=.861281, f1=.820398, auroc=.979100, loss=.943885, minutes=22.4)}
COL = {'iid_plaintext': '#2563eb', 'iid_ckks': '#93c5fd', 'non_iid_plaintext': '#c2410c', 'non_iid_ckks': '#fdba74'}
LBL = {'iid_plaintext': 'IID · plaintext', 'iid_ckks': 'IID · CKKS', 'non_iid_plaintext': 'Non-IID · plaintext', 'non_iid_ckks': 'Non-IID · CKKS'}
esc = html.escape


def mb(b): return f'{b / 1e6:,.2f} MB'
def pct(x): return f'{100 * x:.2f}%'


def line_chart(series, title, ylab, log=False, w=640, h=300, dash=None, fmt=lambda v: f'{v:.3g}'):
    pad_l, pad_r, pad_t, pad_b = 62, 16, 28, 38
    pts = [v for _, s, _ in series for v in s if v is not None and (not log or v > 0)]
    if not pts:
        return ''
    lo, hi = min(pts), max(pts)
    if log:
        lo, hi = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
        if lo == hi: hi += 1
    elif hi == lo:
        hi = lo + 1
    else:
        span = hi - lo; lo -= .05 * span; hi += .05 * span
    n = max(len(s) for _, s, _ in series)
    X = lambda i: pad_l + (w - pad_l - pad_r) * (i / max(1, n - 1))
    Y = lambda v: pad_t + (h - pad_t - pad_b) * (1 - (((math.log10(v) if log else v) - lo) / (hi - lo)))
    g = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(title)}">',
         f'<text x="{pad_l}" y="16" class="ct">{esc(title)}</text>']
    ticks = [10 ** k for k in range(int(lo), int(hi) + 1)] if log else [lo + (hi - lo) * k / 4 for k in range(5)]
    for t in ticks:
        y = Y(t)
        g.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                 f'<text x="{pad_l - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{(f"1e{int(math.log10(t))}" if log else fmt(t))}</text>')
    for i in range(n):
        if n <= 20 and (i == 0 or (i + 1) % 5 == 0):
            g.append(f'<text x="{X(i):.1f}" y="{h - pad_b + 16}" class="tick" text-anchor="middle">{i + 1}</text>')
    g.append(f'<text x="{(w + pad_l) / 2}" y="{h - 6}" class="tick" text-anchor="middle">Communication round</text>'
             f'<text x="14" y="{(h) / 2}" class="tick" transform="rotate(-90 14 {h / 2})" text-anchor="middle">{esc(ylab)}</text>')
    for name, s, color in series:
        d = ' '.join(f'{"M" if j == 0 else "L"}{X(j):.1f},{Y(v):.1f}' for j, v in enumerate(s) if v is not None and (not log or v > 0))
        style = ' stroke-dasharray="6 4"' if dash and name in dash else ''
        g.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"{style}><title>{esc(name)}</title></path>')
        for j, v in enumerate(s):
            if v is not None and (not log or v > 0):
                g.append(f'<circle cx="{X(j):.1f}" cy="{Y(v):.1f}" r="2.6" fill="{color}"><title>{esc(name)} · round {j + 1}: {fmt(v)}</title></circle>')
    g.append('</svg>')
    leg = ''.join(f'<span class="lg"><i style="background:{c}"></i>{esc(nm)}</span>' for nm, _, c in series)
    return f'<figure>{"".join(g)}<figcaption>{leg}</figcaption></figure>'


def hbar(items, title, unit_fmt, w=640):
    rowh = 30; h = 34 + rowh * len(items)
    mx = max(v for _, v, _ in items) or 1
    g = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(title)}"><text x="8" y="16" class="ct">{esc(title)}</text>']
    for i, (name, v, c) in enumerate(items):
        y = 28 + i * rowh; bw = (w - 260) * v / mx
        g.append(f'<text x="8" y="{y + 15}" class="tick">{esc(name)}</text><rect x="170" y="{y + 3}" width="{bw:.1f}" height="18" rx="3" fill="{c}"/>'
                 f'<text x="{176 + bw:.1f}" y="{y + 16}" class="tick">{esc(unit_fmt(v))}</text>')
    g.append('</svg>')
    return f'<figure>{"".join(g)}</figure>'


def stacked(rows, title, w=640):
    parts = [('Encode + encrypt (clients)', 'mean_he_encryption_time_seconds', '#7c3aed'),
             ('Serialize / deserialize', 'mean_he_serialization_time_seconds', '#a78bfa'),
             ('Ciphertext addition (server)', 'mean_he_aggregation_time_seconds', '#059669'),
             ('Decrypt + decode (3rd party)', 'mean_he_decryption_time_seconds', '#f59e0b'),
             ('Packing', 'mean_he_packaging_time_seconds', '#94a3b8')]
    rowh = 34; h = 36 + rowh * len(rows)
    mx = max(sum(r[k] for _, k, _ in parts) for _, r in rows) or 1
    g = [f'<svg viewBox="0 0 {w} {h}" class="chart"><text x="8" y="16" class="ct">{esc(title)}</text>']
    for i, (name, r) in enumerate(rows):
        y = 30 + i * rowh; x = 150
        g.append(f'<text x="8" y="{y + 16}" class="tick">{esc(name)}</text>')
        for pn, k, c in parts:
            bw = (w - 230) * r[k] / mx
            g.append(f'<rect x="{x:.1f}" y="{y + 3}" width="{bw:.1f}" height="20" fill="{c}"><title>{esc(pn)}: {r[k]:.3f} s</title></rect>')
            x += bw
        g.append(f'<text x="{x + 6:.1f}" y="{y + 18}" class="tick">{sum(r[k] for _, k, _ in parts):.2f} s</text>')
    g.append('</svg>')
    leg = ''.join(f'<span class="lg"><i style="background:{c};height:10px;width:10px"></i>{esc(pn)}</span>' for pn, _, c in parts)
    return f'<figure>{"".join(g)}<figcaption>{leg}</figcaption></figure>'


# ---------------- content ----------------
cards = []
for dist in ('iid', 'non_iid'):
    if dist in pairs:
        p = pairs[dist]
        cards.append(f'<div class="card"><div class="k">{"IID" if dist == "iid" else "Non-IID"} · test-accuracy change</div><div class="v">{p["delta_test_accuracy_pp"]:+.3f} pp</div><div class="s">CKKS minus plaintext, same seed &amp; partition</div></div>')
if 'iid' in pairs:
    p = pairs['iid']
    cards.append(f'<div class="card"><div class="k">Added wall time per round</div><div class="v">+{p["added_wall_s_per_round"]:.2f} s</div><div class="s">{p["wall_overhead_pct"]:+.1f}% of the federated phase (IID)</div></div>')
    cards.append(f'<div class="card"><div class="k">Upload per client per round</div><div class="v">{p["upload_expansion_x"]:.1f}×</div><div class="s">ciphertext vs float32 model · total traffic {p["total_traffic_expansion_x"]:.1f}×</div></div>')

rows = []
for dist in ('iid', 'non_iid'):
    b = BASE[dist]
    rows.append(f'<tr class="ref"><td>{esc(b["name"])}</td><td>{pct(b["acc"])}</td><td>{pct(b["f1"])}</td><td>{b["auroc"]:.4f}</td><td>{b["loss"]:.3f}</td><td>{b["minutes"]:.1f} min (GPU)</td><td>–</td><td>{mb(5 * 241993 * 4)}</td><td>{mb(2 * 5 * 241993 * 4)}</td></tr>')
    for mode in ('plaintext', 'ckks'):
        k = f'{dist}_{mode}'
        if k in runs:
            r = runs[k]
            rows.append(f'<tr><td><i class="dot" style="background:{COL[k]}"></i>{LBL[k]}</td><td>{pct(r["test_accuracy"])}</td><td>{pct(r["test_macro_f1"])}</td>'
                        f'<td>{r["test_auroc"]:.4f}</td><td>{r["test_loss"]:.3f}</td><td>{r["federated_wall_s"] / 60:.1f} min</td>'
                        f'<td>{r["federated_cpu_s"] / 60:.1f} min</td><td>{mb(r["upload_bytes_per_round"])}</td><td>{mb(r["total_bytes_per_round"])}</td></tr>')

delta_rows = ''.join(
    f'<tr><td>{"IID" if d == "iid" else "Non-IID"}</td><td>{p["delta_test_accuracy_pp"]:+.4f}</td><td>{p["delta_test_macro_f1_pp"]:+.4f}</td>'
    f'<td>{p["delta_test_auroc"]:+.2e}</td><td>{p["delta_test_loss"]:+.2e}</td><td>{p["max_abs_val_acc_diff_pp_any_round"]:.3f}</td>'
    f'<td>+{p["added_wall_s_per_round"]:.2f} s ({p["wall_overhead_pct"]:+.1f}%)</td><td>+{p["added_cpu_s_per_round"]:.2f} s ({p["cpu_overhead_pct"]:+.1f}%)</td>'
    f'<td>{p["upload_expansion_x"]:.1f}×</td><td>{p["total_traffic_expansion_x"]:.1f}×</td><td>{p["extra_peak_rss_mb"]:+.0f} MB</td></tr>'
    for d, p in pairs.items())

acc_chart = line_chart([(LBL[k], [100 * v for v in runs[k]['val_acc_curve']], COL[k]) for k in COL if k in runs],
                       'Global validation accuracy per round', 'Validation accuracy (%)', dash={LBL['iid_ckks'], LBL['non_iid_ckks']}, fmt=lambda v: f'{v:.2f}')
loss_chart = line_chart([(LBL[k], runs[k]['val_loss_curve'], COL[k]) for k in COL if k in runs],
                        'Global validation loss per round', 'Validation loss', dash={LBL['iid_ckks'], LBL['non_iid_ckks']}, fmt=lambda v: f'{v:.3f}')
err_chart = line_chart([(LBL[k], runs[k]['per_round_max_abs_err'], COL[k.replace('ckks', 'plaintext')]) for k in ('iid_ckks', 'non_iid_ckks') if k in runs],
                       'CKKS aggregate vs plaintext FedAvg on the same client models (max |error|)', 'max |w_CKKS − w_plain|', log=True, w=1040, h=280, fmt=lambda v: f'{v:.2e}')
time_chart = line_chart([(LBL[k], runs[k]['round_wall_curve'], COL[k]) for k in COL if k in runs],
                        'Wall-clock time per round', 'Seconds', dash={LBL['iid_ckks'], LBL['non_iid_ckks']}, fmt=lambda v: f'{v:.1f}')
stack = stacked([(LBL[k], runs[k]) for k in ('iid_ckks', 'non_iid_ckks') if k in runs], 'Where the CKKS time goes (mean per round, all 5 clients + server + decryptor)')
bw_items = []
if 'iid_plaintext' in runs and 'iid_ckks' in runs:
    p, c = runs['iid_plaintext'], runs['iid_ckks']
    bw_items = [('Plain: clients→server', p['upload_bytes_per_round'], '#2563eb'),
                ('Plain: server→clients', p['download_bytes_per_round'], '#60a5fa'),
                ('CKKS: clients→server', c['upload_bytes_per_round'], '#7c3aed'),
                ('CKKS: server→decryptor', c['server_to_decryptor_bytes_per_round'], '#a78bfa'),
                ('CKKS: decryptor→server', c['decryptor_to_server_bytes_per_round'], '#c4b5fd'),
                ('CKKS: server→clients', c['download_bytes_per_round'], '#ddd6fe')]
bw_chart = hbar(bw_items, 'Bytes per round by link (5 clients)', mb, w=1040) if bw_items else ''

scal = sorted([b for b in bench.values() if b['label'] == 'client_scaling'], key=lambda b: b['clients'])
scal_rows = ''.join(f'<tr><td>{b["clients"]}</td><td>{b["ckks_round_wall_s"]:.2f} ± {b["ckks_round_wall_s_std"]:.2f} s</td><td>{b["ckks_round_cpu_s"]:.2f} s</td>'
                    f'<td>{b["per_client_encrypt_wall_s"]:.2f} s</td><td>{b["server_plus_ttp_wall_s"]:.2f} s</td><td>{b["plaintext_agg_wall_s"] * 1e3:.1f} ms</td>'
                    f'<td>{mb(b["he_ciphertext_upload_bytes"])}</td><td>{mb(b["plaintext_upload_bytes"])}</td><td>{b["max_abs_err"]:.1e}</td></tr>' for b in scal)
abl = [b for b in bench.values() if b['label'] == 'param_ablation']
abl_rows = ''.join(f'<tr><td>N={b["params"].split(":")[0]}, q={b["params"].split(":")[1]}, Δ=2^{b["params"].split(":")[2]}</td><td>{b["ckks_round_wall_s"]:.2f} s</td>'
                   f'<td>{mb(b["he_ciphertext_upload_bytes"] / 5)}</td><td>{int(b["he_ciphertext_count"] / 5)}</td><td>{b["he_public_context_bytes"] / 1e3:.0f} KB</td><td>{b["max_abs_err"]:.1e}</td></tr>'
                   for b in sorted(abl, key=lambda b: b['he_ciphertext_upload_bytes'], reverse=True))
shm = [b for b in bench.values() if b['label'] == 'ramdisk_serialization']
ser_note = ''
base5 = [b for b in scal if b['clients'] == 5]
if shm and base5:
    ser_note = (f'<p>Serialization uses SEAL\'s file-based <code>save/load</code> through temporary files. Moving those temporary files to RAM (<code>TMPDIR=/dev/shm</code>) changed serialization from '
                f'{base5[0]["he_serialization_time_seconds"]:.2f} s to {shm[0]["he_serialization_time_seconds"]:.2f} s per 5-client round on this Linux machine. '
                'Your earlier Windows/OneDrive measurement (12.3 s of 14.1 s) is therefore mostly a file-system artefact, not a CKKS cost.</p>')

r0 = next(iter(runs.values()))
meta = (f'5 clients · 20 rounds · 5 local epochs · batch 128 · SGD lr 0.01, momentum 0.9 · Dirichlet α = 0.5 for non-IID · seed 42 · '
        f'device: {esc(str(S.get("device", "see runs")))}')

doc = f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PathMNIST FL — plaintext FedAvg vs CKKS (SEAL) with a third-party decryptor</title>
<style>
:root{{--fg:#0f172a;--mut:#64748b;--bg:#ffffff;--soft:#f1f5f9;--line:#e2e8f0;--acc:#7c3aed}}
@media (prefers-color-scheme:dark){{:root{{--fg:#e2e8f0;--mut:#94a3b8;--bg:#0b1120;--soft:#111a2e;--line:#1e293b}}}}
*{{box-sizing:border-box}}body{{margin:0;font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);background:var(--bg)}}
main{{max-width:1080px;margin:0 auto;padding:28px 20px 60px}}h1{{font-size:26px;margin:0 0 6px}}h2{{font-size:19px;margin:36px 0 10px;border-top:1px solid var(--line);padding-top:22px}}
.sub{{color:var(--mut)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:18px 0}}
.card{{background:var(--soft);border-radius:12px;padding:14px 16px}}.k{{color:var(--mut);font-size:13px}}.v{{font-size:26px;font-weight:700;margin:2px 0}}.s{{color:var(--mut);font-size:12.5px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0}}th,td{{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}
th{{color:var(--mut);font-weight:600}}tr.ref td{{color:var(--mut);font-style:italic}}.wrap{{overflow-x:auto}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
.chart{{width:100%;height:auto}}.ct{{font-size:13px;font-weight:600;fill:var(--fg)}}.tick{{font-size:11px;fill:var(--mut)}}.grid{{stroke:var(--line)}}
figure{{margin:14px 0;background:var(--soft);border-radius:12px;padding:12px}}figcaption{{font-size:12.5px;color:var(--mut)}}.lg{{margin-right:14px}}.lg i{{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:12px}}.note{{background:var(--soft);border-left:3px solid var(--acc);padding:10px 14px;border-radius:6px}}
code{{background:var(--soft);padding:1px 5px;border-radius:4px}}
</style></head><body><main>
<h1>Plaintext FedAvg vs CKKS-encrypted FedAvg on PathMNIST</h1>
<div class="sub">Microsoft SEAL CKKS (via TenSEAL 0.3.18) · clients encrypt, the server adds ciphertexts with only the public key, an independent third-party decryptor holds the secret key and releases only the averaged model.<br>{meta}</div>
<div class="cards">{"".join(cards)}</div>

<h2>1 · Accuracy: does encryption change the model?</h2>
<div class="wrap"><table><tr><th>Run</th><th>Test acc.</th><th>Macro F1</th><th>AUROC</th><th>Test loss</th><th>FL wall time</th><th>FL CPU time</th><th>Upload / round</th><th>Total traffic / round</th></tr>{"".join(rows)}</table></div>
<p class="sub">Italic rows are the plaintext GPU baseline from the 10-experiment report (same hyperparameters), for reference only: they ran on different hardware, so compare accuracy against them and time only within the paired rows.</p>
<div class="grid2">{acc_chart}{loss_chart}</div>
{err_chart}
<div class="wrap"><table><tr><th>Pair</th><th>Δ test acc (pp)</th><th>Δ macro F1 (pp)</th><th>Δ AUROC</th><th>Δ test loss</th><th>max |Δ val acc| any round (pp)</th><th>Added wall / round</th><th>Added CPU / round</th><th>Upload ×</th><th>Total traffic ×</th><th>Extra peak RAM</th></tr>{delta_rows}</table></div>

<h2>2 · Time and computational cost</h2>
<div class="grid2">{time_chart}{stack}</div>
<h3>Scaling with the number of clients (real 241,993-parameter CNN, one aggregation round)</h3>
<div class="wrap"><table><tr><th>Clients</th><th>CKKS round (wall)</th><th>CKKS round (CPU)</th><th>Encrypt per client</th><th>Server add + decrypt</th><th>Plaintext FedAvg</th><th>Ciphertext upload</th><th>Plaintext upload</th><th>Max error</th></tr>{scal_rows}</table></div>
{ser_note}

<h2>3 · Bandwidth</h2>
{bw_chart}
<p>Each client's model (241,993 values) is packed into 60 ciphertexts of 4,096 slots. A fresh ciphertext holds two polynomials over three 40–60-bit primes, so the serialized upload is ≈19.9 MB per client per round versus 0.97 MB of float32 weights (≈20.5×). The public context (parameters + public key, ≈465 KB) is sent once per client, not every round.</p>
<h3>CKKS parameter ablation (5 clients) — addition-only aggregation needs no multiplicative depth</h3>
<div class="wrap"><table><tr><th>Parameters</th><th>Round time</th><th>Upload per client</th><th>Ciphertexts per client</th><th>Public context</th><th>Max error</th></tr>{abl_rows}</table></div>

<h2>4 · How to read this</h2>
<div class="note"><b>What CKKS protects here:</b> the aggregation server never sees an individual hospital's model, only ciphertexts. It cannot decrypt because the secret key is held by the third-party decryptor, which only receives the <i>sum</i>. <b>What it does not protect:</b> the released global model, a colluding server + decryptor, malicious clients (poisoning), or traffic analysis. Local training is unchanged plaintext PyTorch.</div>
<p class="sub">Timings come from one seed on one machine; the plaintext and CKKS twins ran under identical conditions. Byte counts are serialized SEAL payloads (simulated traffic, no network). The global-model download is counted as plaintext in both designs.</p>
</main></body></html>'''
OUT.write_text(doc, encoding='utf-8')
print('wrote', OUT, len(doc))
