"""
Evaluates C4PS's captioning quality on Flickr8k, and tests the paper's core
claim that adaptively enhancing before captioning changes caption quality.

For each sampled Flickr8k image we generate a caption in 5 conditions:
  - original         : GIT captions the raw image directly (no enhancement)
  - fast_x2          : enhance_image(mode='fast')        -> GIT
  - sharp_anime_x4   : enhance_image(mode='sharp_anime') -> GIT
  - general_x4plus   : enhance_image(mode='general')     -> GIT
  - auto_vehicle_x4  : enhance_image(mode='auto_vehicle')-> GIT

"original" corresponds to main.py's Flow A (caption before/independent of
enhancement, used by the general/sharp_anime/auto_vehicle paths); the
enhanced-then-captioned conditions correspond to Flow B (used by the fast
path, where captioning runs on the enhanced image). Comparing these five
conditions against the same Flickr8k references is the direct evidence for
whether C4PS's enhancement step helps or hurts captioning, and for how much
the four enhancement modes differ from each other.

Usage:
    python -m evaluation.evaluate_captioning --n-samples 100 --tile-size 400
"""
import argparse
import csv
import json
import os
import time
import traceback

from . import common  # noqa: E402  (must be imported first — see common.py)
from .common import get_device, results_path, set_all_seeds, free_gpu_memory
from . import datasets
from . import metrics

METHODS = ["original", "fast_x2", "sharp_anime_x4", "general_x4plus", "auto_vehicle_x4"]
MODE_FOR_METHOD = {
    "fast_x2": "fast",
    "sharp_anime_x4": "sharp_anime",
    "general_x4plus": "general",
    "auto_vehicle_x4": "auto_vehicle",
}


def run(n_samples, tile_size, seed, split, resume):
    set_all_seeds(seed)
    device = get_device()
    print(f"[captioning] device={device}")

    from captioning.generator import CaptionGenerator
    from enhancement.enhancer import enhance_image

    caption_model = CaptionGenerator(device)

    samples = datasets.load_flickr8k(n_samples=n_samples, split=split, seed=seed)

    raw_path = results_path("captions_raw.csv")
    fieldnames = ["image_id", "references"] + METHODS
    done_ids = set()
    if resume and os.path.exists(raw_path):
        with open(raw_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_ids.add(row["image_id"])
        print(f"[captioning] resuming: {len(done_ids)} images already done")

    write_header = not (resume and os.path.exists(raw_path))
    f_out = open(raw_path, "a" if resume else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    tmp_image_path = results_path("_tmp_caption_input.png")

    for i, sample in enumerate(samples):
        image_id = sample["image_id"]
        if image_id in done_ids:
            continue
        print(f"[captioning] ({i+1}/{len(samples)}) {image_id}")
        row = {"image_id": image_id, "references": json.dumps(sample["references"])}
        try:
            original_image = sample["image"]
            original_image.save(tmp_image_path)

            row["original"] = caption_model.generate_caption(original_image)

            for method, mode in MODE_FOR_METHOD.items():
                t0 = time.time()
                enhanced = enhance_image(tmp_image_path, mode=mode, tile_size=tile_size, enhance_faces=False)
                caption = caption_model.generate_caption(enhanced)
                row[method] = caption
                print(f"    {method}: \"{caption}\" ({time.time()-t0:.1f}s)")
                free_gpu_memory()

        except Exception as e:
            print(f"[captioning] ERROR on {image_id}: {e}")
            traceback.print_exc()
            for method in METHODS:
                row.setdefault(method, "")

        writer.writerow(row)
        f_out.flush()

    f_out.close()
    if os.path.exists(tmp_image_path):
        os.remove(tmp_image_path)

    compute_metrics_from_csv(raw_path)


def compute_metrics_from_csv(raw_path=None):
    raw_path = raw_path or results_path("captions_raw.csv")
    rows = []
    with open(raw_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    references_by_id = {r["image_id"]: json.loads(r["references"]) for r in rows}

    metrics_rows = []
    for method in ["original"] + list(MODE_FOR_METHOD.keys()):
        candidates_by_id = {r["image_id"]: r[method] for r in rows if r.get(method)}
        if not candidates_by_id:
            continue
        print(f"[captioning] scoring method={method} on {len(candidates_by_id)} images")
        refs_subset = {k: references_by_id[k] for k in candidates_by_id}
        m = metrics.caption_metrics_for_method(candidates_by_id, refs_subset)
        m["method"] = method
        metrics_rows.append(m)

    out_path = results_path("captioning_metrics.csv")
    fieldnames = ["method", "n", "bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rougeL", "cider"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in metrics_rows:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"[captioning] wrote {out_path}")
    for r in metrics_rows:
        print(f"  {r['method']:16s} n={r['n']:3d}  BLEU-4={r['bleu4']:.4f}  METEOR={r['meteor']:.4f}  "
              f"ROUGE-L={r['rougeL']:.4f}  CIDEr={r['cider']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--tile-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--metrics-only", action="store_true",
                         help="Skip generation; just (re)score an existing captions_raw.csv")
    args = parser.parse_args()

    if args.metrics_only:
        compute_metrics_from_csv()
    else:
        run(args.n_samples, args.tile_size, args.seed, args.split, args.resume)


if __name__ == "__main__":
    main()
