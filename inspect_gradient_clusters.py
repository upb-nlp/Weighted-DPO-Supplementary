"""
Read the actual EXAMPLES inside each chosen-gradient cluster, as one
self-contained HTML file.

cluster_chosen_gradients.py says how the clusters sit geometrically; it cannot say
what they MEAN. This script answers that by pulling the dataset text for each
cluster's most representative examples, alongside the statistics that expose the
boring explanations first.

Because the sweep found separation DECREASING monotonically with K -- a continuum,
not modes -- the first question is not "what is cluster 3 about" but "is this
partition tracking anything other than length or gradient norm". So the report
leads with three effect sizes (one-way eta^2, the share of a variable's variance
explained by the cluster label):

    eta^2(log completion length)   how much of length variation the clusters explain
    eta^2(log ||g||)               same for the full-space chosen-gradient norm
    eta^2(log prompt length)

eta^2 near 0 means the clusters are orthogonal to that variable; eta^2 above ~0.3
means the partition largely IS that variable, and "cluster" is a misleading name
for it. Two more views follow:

  - DISTINCTIVE TERMS per cluster: log-odds of each word in the cluster against
    the whole corpus (add-k smoothed), which is what makes a topic legible when
    the centroid geometry does not.
  - A PC1/PC2 SCATTER of a subsample coloured by cluster. On a continuum this
    looks like one blob with colour gradients rather than separated islands --
    the visual counterpart of the flat PC spectrum (PC1 = 4.7%).

Representative examples per cluster are the MEDOIDS (highest cosine to their own
centroid) plus a random sample, so you see both the cluster's core and its bulk.

Preprocessing is mirrored exactly from the clustering run: if that run wrote
global_mean_proj.npy (CENTER = True), the same mean is subtracted here before
normalizing, otherwise the assignments would not reproduce. The script verifies
this by recomputing argmax assignments and reporting agreement with the saved
labels -- anything below 100% means the transforms have diverged.

Inputs:  EMB_DIR (emb_*.npz)  +  CLUSTER_DIR/<WEIGHTING>/K<K>/{assignments,idx,centroids_proj}.npy
Output:  OUT_HTML  (single file, no external assets)  +  a console summary
"""

import html
import json
import os
import re
from collections import Counter

import numpy as np
from dotenv import load_dotenv
load_dotenv(".env")
from datasets import load_dataset

import pissa_lora_common as C

# ── Config ──────────────────────────────────────────────────────────────────
EMB_DIR = "/data/weighted-dpo/pissa-lora-chosen-embeddings"
CLUSTER_DIR = "/data/weighted-dpo/pissa-lora-chosen-clusters"
OUT_HTML = "gradient_clusters.html"

WEIGHTING = "weighted_norm1"    # which emb_<w> / cluster run to inspect
K = 8                           # which K<K>/ directory to read

N_MEDOIDS = 10                  # most-central examples shown per cluster
N_RANDOM = 6                    # additional random members shown per cluster
TEXT_CHARS = 1200               # per-field truncation in the HTML
STATS_SAMPLE = 40_000           # examples read for length/term stats (0 -> all)
SCATTER_POINTS = 4000           # points in the PC1/PC2 scatter
TOP_TERMS = 12                  # distinctive terms listed per cluster
MIN_TERM_COUNT = 15             # ignore terms rarer than this inside a cluster
SEED = 0

STOPWORDS = set("""a an the and or but if then than that this these those of to in on for with as at by
from is are was were be been being it its i you he she they we me him her them my your his their our
not no do does did doing done have has had having will would can could should may might must s t
what which who whom when where why how all any both each few more most other some such only own same
so too very just about into through during before after above below up down out off over under again
there here also very""".split())
TOKEN_RE = re.compile(r"[a-z][a-z']+")


# ── Load ────────────────────────────────────────────────────────────────────

def load_embeddings(emb_dir, weighting):
    """(X [N, dim] fp32, idx [N] int64) from every finalized shard."""
    shards = sorted(f for f in os.listdir(emb_dir)
                    if f.startswith("emb_") and f.endswith(".npz")
                    and not f.endswith(".partial.npz"))
    if not shards:
        raise FileNotFoundError(f"No emb_*.npz in {emb_dir}")
    key = f"emb_{weighting}"
    Xs, idxs, gs = [], [], []
    for name in shards:
        z = np.load(os.path.join(emb_dir, name))
        Xs.append(z[key]); idxs.append(z["idx"]); gs.append(z[f"gnorm_{weighting}"])
    X = Xs[0] if len(Xs) == 1 else np.concatenate(Xs)
    idx = idxs[0] if len(idxs) == 1 else np.concatenate(idxs)
    gnorm = gs[0] if len(gs) == 1 else np.concatenate(gs)
    return X, idx, gnorm


def align_to_clusters(X, emb_idx, gnorm, cl_idx):
    """Reorder embedding rows to match the clustering's saved idx order.

    Done by index lookup rather than by assuming both were sorted the same way,
    so a re-chunked or re-sharded EMB_DIR cannot silently mis-pair rows with
    labels."""
    order = np.argsort(emb_idx)
    # Clip before indexing: a clustered index larger than every embedded index
    # makes searchsorted return len(order), which would raise a bare IndexError
    # instead of the diagnosis below.
    loc = np.clip(np.searchsorted(emb_idx[order], cl_idx), 0, len(order) - 1)
    pos = order[loc]
    if not np.array_equal(emb_idx[pos], cl_idx):
        missing = int((emb_idx[pos] != cl_idx).sum())
        raise RuntimeError(
            f"{missing} clustered indices are absent from {EMB_DIR} — the "
            f"embeddings were rebuilt after clustering. Re-run "
            f"cluster_chosen_gradients.py.")
    return X[pos], gnorm[pos]


def prepare(X, run_dir):
    """Apply the clustering run's exact preprocessing -> unit-norm rows."""
    mean_path = os.path.join(run_dir, "global_mean_proj.npy")
    centered = os.path.isfile(mean_path)
    if centered:
        X = X - np.load(mean_path)
        print(f"  CENTER=True run: subtracted {os.path.basename(mean_path)}")
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12), centered


# ── Statistics ──────────────────────────────────────────────────────────────

def eta_squared(values, labels, K):
    """One-way eta^2: share of `values` variance explained by the cluster label.

    0 = clusters say nothing about this variable; 1 = the clusters ARE this
    variable. Reported for log length and log ||g|| because those are the
    explanations that would make the clustering uninteresting."""
    values = np.asarray(values, dtype=np.float64)
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for k in range(K):
        v = values[labels == k]
        if len(v):
            ss_between += len(v) * (v.mean() - grand) ** 2
    return float(ss_between / ss_total)


def distinctive_terms(texts_by_cluster, K):
    """Top terms per cluster by smoothed log-odds against the whole corpus."""
    per = [Counter() for _ in range(K)]
    total = Counter()
    for k, texts in enumerate(texts_by_cluster):
        for t in texts:
            for w in TOKEN_RE.findall(t.lower()):
                if w not in STOPWORDS and len(w) > 2:
                    per[k][w] += 1
                    total[w] += 1
    N_all = sum(total.values()) or 1
    V = len(total) or 1
    out = []
    for k in range(K):
        N_k = sum(per[k].values()) or 1
        scored = []
        for w, c in per[k].items():
            if c < MIN_TERM_COUNT:
                continue
            p_k = (c + 1.0) / (N_k + V)
            p_all = (total[w] + 1.0) / (N_all + V)
            scored.append((np.log(p_k / p_all), w, c))
        scored.sort(reverse=True)
        out.append([{"term": w, "count": int(c), "logodds": round(float(s), 3)}
                    for s, w, c in scored[:TOP_TERMS]])
    return out


def pca_2d(Xn, rng, n_points):
    """PC1/PC2 coordinates for a subsample (covariance is only 1000x1000)."""
    take = rng.choice(len(Xn), size=min(n_points, len(Xn)), replace=False)
    sub = Xn[take]
    mu = Xn.mean(0)
    Xc = sub - mu
    cov = (Xn - mu).T @ (Xn - mu) / len(Xn)
    w, V = np.linalg.eigh(cov.astype(np.float64))
    pcs = V[:, ::-1][:, :2]
    return take, (Xc @ pcs).astype(np.float32), (w[::-1][:2] / w.sum()).tolist()


# ── Text ────────────────────────────────────────────────────────────────────

def message_texts(example):
    """(prompt, chosen completion, rejected completion) as plain strings."""
    def last(msgs, role=None):
        for m in reversed(msgs or []):
            if role is None or m.get("role") == role:
                return (m.get("content") or "")
        return ""
    chosen, rejected = example.get("chosen") or [], example.get("rejected") or []
    return last(chosen, "user"), last(chosen), last(rejected)


# ── HTML ────────────────────────────────────────────────────────────────────

TEMPLATE = """<!doctype html>
<meta charset="utf-8"><title>Chosen-gradient clusters</title>
<style>
:root{--bg:#fff;--fg:#111;--mut:#666;--line:#ddd;--card:#fafafa;--acc:#0b5;}
@media(prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8e8e8;--mut:#9aa;--line:#333;--card:#1b1e24;--acc:#5d9;}}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg)}
header{padding:14px 18px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
h1{font-size:16px;margin:0 0 6px}
.mut{color:var(--mut)}
.warn{color:#c40;font-weight:600}
main{padding:14px 18px;max-width:1200px}
table{border-collapse:collapse;font-size:13px;width:100%;overflow-x:auto;display:block}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{background:var(--card)}
tr.sel{outline:2px solid var(--acc)}
.eta{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0}
.eta div{background:var(--card);border:1px solid var(--line);padding:8px 12px;border-radius:6px}
.eta b{font-size:18px;display:block}
.terms{font-size:12px;color:var(--mut)}
.ex{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px;margin:10px 0}
.ex h4{margin:0 0 6px;font-size:13px}
.ex pre{white-space:pre-wrap;word-break:break-word;margin:4px 0;font:12px/1.45 ui-monospace,Menlo,monospace;
        max-height:16em;overflow:auto;background:var(--bg);border:1px solid var(--line);padding:6px;border-radius:4px}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
.badge{display:inline-block;padding:1px 6px;border-radius:10px;background:var(--line);font-size:11px;margin-right:6px}
svg{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:var(--card)}
button{font:inherit;padding:4px 10px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:5px;cursor:pointer}
</style>
<header>
  <h1>Chosen-gradient clusters &middot; <span id="hk"></span></h1>
  <div class="mut" id="hmeta"></div>
</header>
<main>
  <div class="eta" id="eta"></div>
  <div id="scatterbox"></div>
  <h3>Clusters <span class="mut" style="font-weight:400">(click a row)</span></h3>
  <table id="tbl"></table>
  <div id="detail"></div>
</main>
<script id="data" type="application/json">/*__DATA__*/</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const f = (x,n=3) => (x==null||Number.isNaN(x)) ? "-" : Number(x).toFixed(n);
const HUE = i => `hsl(${Math.round(360*i/Math.max(D.clusters.length,1))} 70% 50%)`;

document.getElementById("hk").textContent =
  `K=${D.meta.K}  ${D.meta.weighting}  ${D.meta.n.toLocaleString()} examples`;
document.getElementById("hmeta").innerHTML =
  `centered=<b>${D.meta.centered}</b> &middot; assignment agreement ` +
  `<b class="${D.meta.agreement<1?'warn':''}">${(100*D.meta.agreement).toFixed(2)}%</b>` +
  ` &middot; stats from ${D.meta.n_stats.toLocaleString()} sampled examples` +
  (D.meta.agreement<1 ? " &middot; <span class='warn'>preprocessing differs from the clustering run</span>" : "");

document.getElementById("eta").innerHTML = D.eta.map(e =>
  `<div><span class="lbl">${e.name}</span><b>${f(e.value)}</b>
   <span class="mut">${e.value>=0.3?"clusters largely ARE this":e.value>=0.1?"partly explained":"mostly independent"}</span></div>`
).join("");

if (D.scatter) {
  const W=760,H=420,P=28, xs=D.scatter.pts.map(p=>p[0]), ys=D.scatter.pts.map(p=>p[1]);
  const xr=[Math.min(...xs),Math.max(...xs)], yr=[Math.min(...ys),Math.max(...ys)];
  const sx=v=>P+(W-2*P)*(v-xr[0])/((xr[1]-xr[0])||1), sy=v=>H-P-(H-2*P)*(v-yr[0])/((yr[1]-yr[0])||1);
  document.getElementById("scatterbox").innerHTML =
    `<h3>PC1 / PC2 of the projected gradients <span class="mut" style="font-weight:400">
      (${D.scatter.var[0]!=null?(100*D.scatter.var[0]).toFixed(1):"?"}% / ${(100*D.scatter.var[1]).toFixed(1)}% of variance;
      separated islands = real modes, one blob = continuum)</span></h3>
     <svg viewBox="0 0 ${W} ${H}" role="img">` +
    D.scatter.pts.map(p=>`<circle cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="1.7"
        fill="${HUE(p[2])}" opacity=".55"/>`).join("") + `</svg>`;
}

const tbl = document.getElementById("tbl");
tbl.innerHTML = "<thead><tr><th>cluster</th><th>size</th><th>share</th><th>cohesion</th>" +
  "<th>nearest</th><th>cos</th><th>med chosen chars</th><th>med ||g||</th><th>top terms</th></tr></thead><tbody>" +
  D.clusters.map(c => `<tr data-k="${c.k}">
     <td><span class="badge" style="background:${HUE(c.k)}">&nbsp;</span>${c.k}</td>
     <td>${c.size.toLocaleString()}</td><td>${(100*c.share).toFixed(1)}%</td>
     <td>${f(c.cohesion)}</td><td>${c.nearest}</td><td>${f(c.nearest_cos)}</td>
     <td>${c.med_chosen_chars ?? "-"}</td><td>${f(c.med_gnorm,2)}</td>
     <td class="terms">${c.terms.map(t=>esc(t.term)).join(", ")}</td></tr>`).join("") + "</tbody>";

function show(k) {
  const c = D.clusters[k];
  [...tbl.querySelectorAll("tr")].forEach(r => r.classList.toggle("sel", r.dataset.k == String(k)));
  const ex = g => g.map(e => `<div class="ex">
      <h4>#${e.idx} <span class="mut">cos to centroid ${f(e.cos)} &middot; ||g|| ${f(e.gnorm,2)}
      &middot; chosen ${e.n_chosen} chars &middot; rejected ${e.n_rejected} chars</span></h4>
      <div class="lbl">prompt</div><pre>${esc(e.prompt)}</pre>
      <div class="lbl">chosen</div><pre>${esc(e.chosen)}</pre>
      <div class="lbl">rejected</div><pre>${esc(e.rejected)}</pre></div>`).join("");
  document.getElementById("detail").innerHTML =
    `<h3>Cluster ${k} <span class="mut" style="font-weight:400">${c.size.toLocaleString()} members</span></h3>
     <div class="terms">distinctive: ${c.terms.map(t=>`${esc(t.term)} <span class="mut">(${t.count}, ${f(t.logodds,2)})</span>`).join(" &middot; ")}</div>
     <h4>Most central (medoids)</h4>${ex(c.medoids)}
     <h4>Random members</h4>${ex(c.random)}`;
  document.getElementById("detail").scrollIntoView({behavior:"smooth", block:"start"});
}
tbl.addEventListener("click", e => { const r = e.target.closest("tr[data-k]"); if (r) show(+r.dataset.k); });
show(0);
</script>
"""


def main():
    run_dir = os.path.join(CLUSTER_DIR, WEIGHTING, f"K{K}")
    for f_ in ("assignments.npy", "idx.npy", "centroids_proj.npy"):
        if not os.path.isfile(os.path.join(run_dir, f_)):
            raise FileNotFoundError(
                f"{os.path.join(run_dir, f_)} missing. Run cluster_chosen_gradients.py "
                f"with K={K} in K_SWEEP and WEIGHTING={WEIGHTING!r} first.")
    labels = np.load(os.path.join(run_dir, "assignments.npy")).astype(np.int64)
    cl_idx = np.load(os.path.join(run_dir, "idx.npy"))
    Cn = np.load(os.path.join(run_dir, "centroids_proj.npy"))
    print(f"Cluster run {run_dir}: {len(labels):,} rows, K={Cn.shape[0]}")

    print(f"Loading embeddings from {EMB_DIR} ...")
    X, emb_idx, gnorm = load_embeddings(EMB_DIR, WEIGHTING)
    X, gnorm = align_to_clusters(X, emb_idx, gnorm, cl_idx)
    del emb_idx
    Xn, centered = prepare(X, run_dir)
    del X

    # Reproduce the assignment; a mismatch means the preprocessing diverged.
    sim = Xn @ Cn.T
    recomputed = sim.argmax(1)
    agreement = float((recomputed == labels).mean())
    own_cos = sim[np.arange(len(labels)), labels]
    print(f"  assignment agreement with saved labels: {100*agreement:.2f}%"
          + ("" if agreement == 1.0 else "   <-- WARNING: transforms differ"))

    # Nearest OTHER centroid per cluster.
    cc = Cn @ Cn.T
    np.fill_diagonal(cc, -2.0)
    nearest = cc.argmax(1)

    print(f"Loading dataset {C.DATASET_NAME} ...")
    dataset = load_dataset(C.DATASET_NAME, split="train")
    rng = np.random.default_rng(SEED)

    # Which rows to read text for: the displayed examples (medoids + random per
    # cluster) plus a sample for the aggregate length/term statistics.
    K_ = Cn.shape[0]
    medoids, randoms = [], []
    for k in range(K_):
        members = np.flatnonzero(labels == k)
        if not len(members):
            medoids.append(np.array([], int)); randoms.append(np.array([], int)); continue
        best = members[np.argsort(-own_cos[members])][:N_MEDOIDS]
        rest = np.setdiff1d(members, best, assume_unique=False)
        pick = rng.choice(rest, size=min(N_RANDOM, len(rest)), replace=False) if len(rest) else np.array([], int)
        medoids.append(best); randoms.append(np.sort(pick))

    n_stats = len(labels) if not STATS_SAMPLE else min(STATS_SAMPLE, len(labels))
    stats_rows = (np.arange(len(labels)) if n_stats == len(labels)
                  else rng.choice(len(labels), size=n_stats, replace=False))
    need = np.unique(np.concatenate([stats_rows] + medoids + randoms).astype(np.int64))
    print(f"Reading text for {len(need):,} examples "
          f"({n_stats:,} for stats, {sum(len(m)+len(r) for m, r in zip(medoids, randoms))} displayed) ...")

    texts = {}
    for row in need:
        texts[int(row)] = message_texts(dataset[int(cl_idx[row])])

    # Aggregate statistics over the sampled rows.
    s_labels = labels[stats_rows]
    prompt_len = np.array([len(texts[int(r)][0]) for r in stats_rows], float)
    chosen_len = np.array([len(texts[int(r)][1]) for r in stats_rows], float)
    log = lambda v: np.log(np.maximum(v, 1.0))
    eta = [
        {"name": "eta^2 log chosen length", "value": eta_squared(log(chosen_len), s_labels, K_)},
        {"name": "eta^2 log ||g||", "value": eta_squared(log(gnorm[stats_rows]), s_labels, K_)},
        {"name": "eta^2 log prompt length", "value": eta_squared(log(prompt_len), s_labels, K_)},
    ]
    print("\n--- how much do the clusters just encode length / norm? ---")
    for e in eta:
        verdict = ("clusters largely ARE this variable" if e["value"] >= 0.3
                   else "partly explained" if e["value"] >= 0.1 else "mostly independent")
        print(f"  {e['name']:26s} {e['value']:.3f}   {verdict}")

    terms = distinctive_terms(
        [[texts[int(r)][1] for r in stats_rows[s_labels == k]] for k in range(K_)], K_)

    take, pts2d, var2 = pca_2d(Xn, np.random.default_rng(SEED), SCATTER_POINTS)
    scatter = {"pts": [[round(float(a), 3), round(float(b), 3), int(labels[t])]
                       for (a, b), t in zip(pts2d, take)],
               "var": [float(var2[0]), float(var2[1])]}

    def pack(rows, k):
        out = []
        for r in rows:
            p, ch, rj = texts[int(r)]
            out.append({"idx": int(cl_idx[r]), "cos": round(float(own_cos[r]), 4),
                        "gnorm": round(float(gnorm[r]), 3),
                        "n_chosen": len(ch), "n_rejected": len(rj),
                        "prompt": p[:TEXT_CHARS], "chosen": ch[:TEXT_CHARS],
                        "rejected": rj[:TEXT_CHARS]})
        return out

    clusters = []
    print("\n--- clusters ---")
    for k in range(K_):
        members = np.flatnonzero(labels == k)
        m_stats = stats_rows[s_labels == k]
        med_chars = int(np.median(chosen_len[s_labels == k])) if len(m_stats) else None
        clusters.append({
            "k": k, "size": int(len(members)), "share": float(len(members) / len(labels)),
            "cohesion": float(own_cos[members].mean()) if len(members) else None,
            "nearest": int(nearest[k]), "nearest_cos": float(cc[k, nearest[k]]),
            "med_chosen_chars": med_chars,
            "med_gnorm": float(np.median(gnorm[members])) if len(members) else None,
            "terms": terms[k],
            "medoids": pack(medoids[k], k), "random": pack(randoms[k], k),
        })
        print(f"  k={k:3d} n={len(members):7,d} ({100*len(members)/len(labels):5.1f}%) "
              f"cohesion={clusters[-1]['cohesion']:.3f} nearest={nearest[k]}"
              f"({cc[k, nearest[k]]:.3f}) med_chars={med_chars} "
              f"terms: {', '.join(t['term'] for t in terms[k][:6])}")

    payload = {"meta": {"K": K_, "weighting": WEIGHTING, "n": int(len(labels)),
                        "n_stats": int(n_stats), "centered": bool(centered),
                        "agreement": agreement, "run_dir": run_dir},
               "eta": eta, "clusters": clusters, "scatter": scatter}
    # Escape "<" as \u003c before embedding: completions in this dataset contain
    # code, so a literal "</script>" in any prompt/chosen/rejected string would
    # close the <script type="application/json"> block early and break the page.
    # \u003c is a valid JSON string escape that JSON.parse decodes back, and JSON
    # has no "<" outside of strings, so a blanket replace is safe. U+2028/2029 are
    # escaped for the same reason (they are raw line terminators in JS).
    blob = (json.dumps(payload).replace("<", "\\u003c")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    with open(OUT_HTML, "w") as fh:
        fh.write(TEMPLATE.replace("/*__DATA__*/", blob))
    print(f"\nWrote {OUT_HTML} ({os.path.getsize(OUT_HTML)/1e6:.1f} MB) — open it in a browser.")
    print("Read the eta^2 row first: if the clusters mostly encode length or ||g||, "
          "K modal directions will not add anything the aggregated mean lacks.")


if __name__ == "__main__":
    main()
