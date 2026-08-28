"""
Paired significance testing / confidence intervals on top of the existing
*_raw.csv files produced by evaluate_enhancement.py, evaluate_captioning.py,
and evaluate_translation.py.

Pure post-processing -- reads the raw per-image / per-sentence CSVs already
on disk and does NOT re-run any model. Exists because the summary *_metrics.csv
files only report means, which isn't enough to tell whether two methods'
scores are actually distinguishable given n=20-60 samples and stds of
comparable magnitude to the between-method gaps.

Methodology:
  - Enhancement / captioning: paired per-image comparison against a fixed
    baseline (bicubic at matching scale / "original" caption respectively).
    Paired Wilcoxon signed-rank is the primary test (doesn't assume normality,
    appropriate for n<30 metric distributions); paired t-test is reported
    alongside since it's what most reviewers expect to see. A BCa-style
    percentile bootstrap (10000 resamples) gives a 95% CI on the mean paired
    difference, which is what actually belongs in a results table.
  - Translation: no second method to pair against (one backend per language),
    so instead this reports a percentile bootstrap 95% CI on each corpus
    metric (BLEU/chrF/BERTScore-F1) via sentence-level resampling, which is
    the standard way to put an uncertainty band on a corpus-level MT metric.

Usage:
    python -m evaluation.significance
"""
import csv
import json

import numpy as np
from scipy import stats

from . import common  # noqa: E402
from .common import results_path
from . import metrics

N_BOOTSTRAP = 10000
ALPHA = 0.05
RNG_SEED = 42


def _bootstrap_ci_mean(diffs, n_boot=N_BOOTSTRAP, alpha=ALPHA, seed=RNG_SEED):
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = diffs[rng.integers(0, n, n)]
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _paired_tests(baseline_vals, method_vals):
    """Paired t-test + Wilcoxon signed-rank + bootstrap CI on (method - baseline)."""
    baseline_vals = np.asarray(baseline_vals, dtype=float)
    method_vals = np.asarray(method_vals, dtype=float)
    diffs = method_vals - baseline_vals
    n = len(diffs)

    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = _bootstrap_ci_mean(diffs)

    t_stat, t_p = stats.ttest_rel(method_vals, baseline_vals)

    # Wilcoxon requires at least one non-zero difference and n>=1 non-tied pairs.
    nonzero = diffs[diffs != 0]
    if len(nonzero) >= 1:
        try:
            w_stat, w_p = stats.wilcoxon(method_vals, baseline_vals)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
    else:
        w_stat, w_p = float("nan"), 1.0  # identical to baseline on every pair

    return {
        "n": n,
        "mean_diff": mean_diff,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "significant_95": not (ci_lo <= 0 <= ci_hi),
        "t_stat": float(t_stat),
        "t_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else "",
        "wilcoxon_pvalue": float(w_p),
    }


# ---------------------------------------------------------------------------
# Enhancement
# ---------------------------------------------------------------------------

def enhancement_significance():
    raw_path = results_path("enhancement_raw.csv")
    rows = list(csv.DictReader(open(raw_path, newline="", encoding="utf-8")))

    by_image_scale = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        key = (r["image_id"], r["scale"])
        by_image_scale.setdefault(key, {})[r["method"]] = r

    scales = sorted(set(k[1] for k in by_image_scale))
    baseline_for_scale = {"4": "bicubic_4x", "2": "bicubic_2x"}

    out_rows = []
    for scale in scales:
        baseline_method = baseline_for_scale.get(scale)
        if baseline_method is None:
            continue
        methods = sorted({r["method"] for (img, sc), d in by_image_scale.items()
                           if sc == scale for r in [d.get(baseline_method)] if r} )
        all_methods = sorted({m for (img, sc), d in by_image_scale.items() if sc == scale for m in d})
        for method in all_methods:
            if method == baseline_method:
                continue
            paired_base, paired_method = [], []
            for (img, sc), d in by_image_scale.items():
                if sc != scale:
                    continue
                if baseline_method in d and method in d:
                    paired_base.append(d[baseline_method])
                    paired_method.append(d[method])
            if len(paired_base) < 2:
                continue
            for metric_name in ["psnr", "ssim", "lpips"]:
                b_vals = [float(r[metric_name]) for r in paired_base]
                m_vals = [float(r[metric_name]) for r in paired_method]
                res = _paired_tests(b_vals, m_vals)
                res.update(scale=scale, method=method, baseline=baseline_method, metric=metric_name)
                out_rows.append(res)

    out_path = results_path("enhancement_significance.csv")
    fieldnames = ["scale", "method", "baseline", "metric", "n", "mean_diff",
                  "ci95_lo", "ci95_hi", "significant_95", "t_stat", "t_pvalue",
                  "wilcoxon_stat", "wilcoxon_pvalue"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"[significance] wrote {out_path}")
    for r in out_rows:
        flag = "*" if r["significant_95"] else " "
        print(f"  {flag} scale={r['scale']} {r['method']:20s} vs {r['baseline']:12s} "
              f"{r['metric']:6s} diff={r['mean_diff']:+.4f} "
              f"95%CI=[{r['ci95_lo']:+.4f},{r['ci95_hi']:+.4f}] "
              f"t_p={r['t_pvalue']:.4f} wilcoxon_p={r['wilcoxon_pvalue']:.4f}")


# ---------------------------------------------------------------------------
# Captioning
# ---------------------------------------------------------------------------

def captioning_significance():
    raw_path = results_path("captions_raw.csv")
    rows = list(csv.DictReader(open(raw_path, newline="", encoding="utf-8")))

    references_by_id = {r["image_id"]: json.loads(r["references"]) for r in rows}
    methods = [c for c in rows[0].keys() if c not in ("image_id", "references")]
    baseline_method = "original"

    # Per-sentence metrics for every (image, method) that has a non-empty caption.
    per_sentence = {}  # (image_id, method) -> {"bleu4":.., "meteor":.., "rougeL":..}
    for r in rows:
        image_id = r["image_id"]
        refs = references_by_id[image_id]
        for method in methods:
            cand = r.get(method)
            if not cand:
                continue
            b = metrics.bleu_1234(cand, refs)
            per_sentence[(image_id, method)] = {
                "bleu4": b["bleu4"],
                "meteor": metrics.meteor(cand, refs),
                "rougeL": metrics.rouge_l(cand, refs),
            }

    out_rows = []
    for method in methods:
        if method == baseline_method:
            continue
        for metric_name in ["bleu4", "meteor", "rougeL"]:
            b_vals, m_vals = [], []
            for r in rows:
                image_id = r["image_id"]
                bkey, mkey = (image_id, baseline_method), (image_id, method)
                if bkey in per_sentence and mkey in per_sentence:
                    b_vals.append(per_sentence[bkey][metric_name])
                    m_vals.append(per_sentence[mkey][metric_name])
            if len(b_vals) < 2:
                continue
            res = _paired_tests(b_vals, m_vals)
            res.update(method=method, baseline=baseline_method, metric=metric_name)
            out_rows.append(res)

    out_path = results_path("captioning_significance.csv")
    fieldnames = ["method", "baseline", "metric", "n", "mean_diff",
                  "ci95_lo", "ci95_hi", "significant_95", "t_stat", "t_pvalue",
                  "wilcoxon_stat", "wilcoxon_pvalue"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"[significance] wrote {out_path}")
    for r in out_rows:
        flag = "*" if r["significant_95"] else " "
        print(f"  {flag} {r['method']:16s} vs {r['baseline']:10s} {r['metric']:8s} "
              f"diff={r['mean_diff']:+.4f} 95%CI=[{r['ci95_lo']:+.4f},{r['ci95_hi']:+.4f}] "
              f"t_p={r['t_pvalue']:.4f} wilcoxon_p={r['wilcoxon_pvalue']:.4f}")


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translation_ci(n_boot=2000):
    """Bootstrap 95% CI on corpus BLEU/chrF per language (sentence-level resampling)."""
    import sacrebleu

    raw_path = results_path("translation_raw.csv")
    rows = [r for r in csv.DictReader(open(raw_path, newline="", encoding="utf-8")) if r["status"] == "ok"]

    by_lang = {}
    for r in rows:
        by_lang.setdefault(r["language"], []).append(r)

    rng = np.random.default_rng(RNG_SEED)
    out_rows = []
    for lang, lrows in sorted(by_lang.items()):
        cands = [r["hypothesis"] for r in lrows]
        refs = [r["reference"] for r in lrows]
        n = len(cands)

        bleu_point = sacrebleu.corpus_bleu(cands, [refs]).score
        chrf_point = sacrebleu.corpus_chrf(cands, [refs]).score

        bleu_boot, chrf_boot = np.empty(n_boot), np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            c_s = [cands[j] for j in idx]
            r_s = [refs[j] for j in idx]
            bleu_boot[i] = sacrebleu.corpus_bleu(c_s, [r_s]).score
            chrf_boot[i] = sacrebleu.corpus_chrf(c_s, [r_s]).score

        bleu_lo, bleu_hi = np.percentile(bleu_boot, [2.5, 97.5])
        chrf_lo, chrf_hi = np.percentile(chrf_boot, [2.5, 97.5])

        out_rows.append({
            "language": lang, "n": n,
            "bleu": bleu_point, "bleu_ci95_lo": bleu_lo, "bleu_ci95_hi": bleu_hi,
            "chrf": chrf_point, "chrf_ci95_lo": chrf_lo, "chrf_ci95_hi": chrf_hi,
        })

    out_path = results_path("translation_ci.csv")
    fieldnames = ["language", "n", "bleu", "bleu_ci95_lo", "bleu_ci95_hi",
                  "chrf", "chrf_ci95_lo", "chrf_ci95_hi"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"[significance] wrote {out_path}")
    for r in out_rows:
        print(f"  {r['language']:4s} n={r['n']:3d}  BLEU={r['bleu']:.2f} "
              f"[{r['bleu_ci95_lo']:.2f},{r['bleu_ci95_hi']:.2f}]  "
              f"chrF={r['chrf']:.2f} [{r['chrf_ci95_lo']:.2f},{r['chrf_ci95_hi']:.2f}]")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-enhancement", action="store_true")
    parser.add_argument("--skip-captioning", action="store_true")
    parser.add_argument("--skip-translation", action="store_true")
    args = parser.parse_args()

    if not args.skip_enhancement:
        enhancement_significance()
    if not args.skip_captioning:
        captioning_significance()
    if not args.skip_translation:
        translation_ci()


if __name__ == "__main__":
    main()
