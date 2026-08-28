"""
Evaluates C4PS's translation quality and semantic preservation, per language,
against FLORES-200 -- a standard sentence-aligned multilingual MT benchmark
(not the Flickr8k captions, which have no human reference translations).

For each target language, English FLORES-200 sentences are passed through
C4PS's own routing logic (translation/router.py -> Indic/NLLB/MarianMT
backend, exactly as main.py uses it) and compared against the FLORES-200
reference translation for the same sentence using BLEU, chrF, and BERTScore
F1 (semantic-similarity metric, addressing "semantic preservation").

Usage:
    python -m evaluation.evaluate_translation --languages core --n-sentences 30
    python -m evaluation.evaluate_translation --languages all --n-sentences 30
"""
import argparse
import csv
import time
import traceback

from . import common  # noqa: E402
from .common import results_path, free_gpu_memory
from . import datasets
from . import metrics
from translation.router import translate_text, INDIC_LANGS, NLLB_LANGS, MARIAN_LANGS

CORE_LANGUAGES = ["hi", "kn", "ta", "te", "es", "fr"]
ALL_LANGUAGES = sorted(INDIC_LANGS | NLLB_LANGS | MARIAN_LANGS)


def backend_for(lang):
    if lang in INDIC_LANGS:
        return "indic_nllb"
    if lang in NLLB_LANGS:
        return "nllb"
    if lang in MARIAN_LANGS:
        return "marian"
    return "unknown"


def run(languages, n_sentences, seed):
    parallel = datasets.load_flores_parallel(languages, n_sentences=n_sentences, seed=seed)
    available_langs = [l for l in languages if l in parallel and l != "en"]
    print(f"[translation] evaluating languages: {available_langs}")

    raw_path = results_path("translation_raw.csv")
    fieldnames = ["language", "backend", "source_en", "reference", "hypothesis", "elapsed_seconds", "status"]
    f_out = open(raw_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for lang in available_langs:
        backend = backend_for(lang)
        print(f"[translation] === {lang} ({backend}) ===")
        for src, ref in zip(parallel["en"], parallel[lang]):
            row = {"language": lang, "backend": backend, "source_en": src, "reference": ref}
            try:
                t0 = time.time()
                hyp = translate_text(src, lang, source_lang="en")
                elapsed = time.time() - t0
                row.update(hypothesis=hyp, elapsed_seconds=elapsed, status="ok")
            except Exception as e:
                print(f"[translation] ERROR translating to {lang}: {e}")
                traceback.print_exc()
                row.update(hypothesis="", elapsed_seconds="", status=f"error: {e}")
            writer.writerow(row)
            f_out.flush()
        free_gpu_memory()

    f_out.close()
    summarize(raw_path)


def summarize(raw_path=None):
    raw_path = raw_path or results_path("translation_raw.csv")
    import pandas as pd
    df = pd.read_csv(raw_path)
    df_ok = df[df["status"] == "ok"].copy()

    rows = []
    for lang, group in df_ok.groupby("language"):
        candidates = group["hypothesis"].astype(str).tolist()
        references = group["reference"].astype(str).tolist()
        if not candidates:
            continue
        print(f"[translation] scoring {lang} (n={len(candidates)})")
        bleu, chrf = metrics.bleu_chrf_corpus(candidates, references)
        try:
            bert_f1 = metrics.bertscore_f1(candidates, references, lang=lang)
        except Exception as e:
            print(f"[translation] BERTScore failed for {lang}: {e}")
            bert_f1 = ""
        rows.append({
            "language": lang,
            "backend": group["backend"].iloc[0],
            "n": len(candidates),
            "bleu": bleu,
            "chrf": chrf,
            "bertscore_f1": bert_f1,
        })

    out_path = results_path("translation_metrics.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "backend", "n", "bleu", "chrf", "bertscore_f1"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[translation] wrote {out_path}")
    for r in rows:
        print(f"  {r['language']:4s} ({r['backend']:10s}) n={r['n']:3d}  BLEU={r['bleu']:.2f}  "
              f"chrF={r['chrf']:.2f}  BERTScore-F1={r['bertscore_f1']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", type=str, default="core",
                         help="'core' (hi,kn,ta,te,es,fr), 'all', or a comma-separated list of codes")
    parser.add_argument("--n-sentences", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if args.languages == "core":
        languages = CORE_LANGUAGES
    elif args.languages == "all":
        languages = ALL_LANGUAGES
    else:
        languages = [l.strip() for l in args.languages.split(",")]

    if args.summarize_only:
        summarize()
    else:
        run(languages, args.n_sentences, args.seed)


if __name__ == "__main__":
    main()
