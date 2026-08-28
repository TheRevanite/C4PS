"""
Runtime benchmark across enhancement mode x resolution x tile size, with
repeats so mean+std can be reported instead of a single anecdotal timing.

Each (mode, resolution, tile_size, image) config is run in a fresh
subprocess with a hard timeout, mirroring tests/sweep_enhancement.py's
approach in this repo -- a config that OOMs or hangs is recorded as such
and does not take down the whole sweep. Progress is appended to CSV as it
goes, so the sweep is resumable and safe to run unattended overnight.

Usage:
    python -m evaluation.benchmark_runtime --n-images 5 --repeats 3
"""
import argparse
import csv
import itertools
import multiprocessing as mp
import os
import time

from . import common  # noqa: E402
from .common import results_path

MODES = {
    "fast_x2": "fast",
    "sharp_anime_x4": "sharp_anime",
    "general_x4plus": "general",
    "auto_vehicle_x4": "auto_vehicle",
    "general_x4plus_gfpgan": "general",  # face-enhance variant of general
}
RESOLUTIONS = [512, 1024, 1920]
TILE_SIZES = [128, 256, 512]


def _worker(image_path, mode, tile_size, enhance_faces, q):
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from evaluation import common as _common  # noqa: F401  (monkeypatch)
        from enhancement.enhancer import enhance_image
        t0 = time.time()
        enhance_image(image_path, mode=mode, tile_size=tile_size, enhance_faces=enhance_faces)
        elapsed = time.time() - t0
        q.put({"status": "ok", "elapsed": elapsed})
    except Exception as e:
        q.put({"status": "error", "error": str(e)})


def make_test_images(n_images, resolutions, seed):
    from . import datasets
    from PIL import Image
    samples = datasets.load_flickr8k(n_samples=n_images, split="train", seed=seed + 12345)
    paths_by_res = {}
    for res in resolutions:
        paths_by_res[res] = []
        for i, sample in enumerate(samples):
            img = sample["image"].convert("RGB")
            w, h = img.size
            scale = res / max(w, h)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            resized = img.resize(new_size, Image.BICUBIC)
            path = results_path("bench_images", f"res{res}_img{i}.png")
            resized.save(path)
            paths_by_res[res].append(path)
    return paths_by_res


def run(n_images, repeats, seed, timeout_general, timeout_other):
    paths_by_res = make_test_images(n_images, RESOLUTIONS, seed)

    out_path = results_path("runtime_raw.csv")
    fieldnames = ["mode", "resolution", "tile_size", "image_idx", "run_idx", "elapsed_seconds", "status"]
    write_header = not os.path.exists(out_path)
    f_out = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    done = set()
    if not write_header:
        with open(out_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["mode"], row["resolution"], row["tile_size"], row["image_idx"], row["run_idx"]))

    combos = list(itertools.product(MODES.items(), RESOLUTIONS, TILE_SIZES, range(n_images), range(repeats)))
    print(f"[runtime] {len(combos)} total (mode,resolution,tile,image,repeat) configs")

    for (method, mode), resolution, tile_size, image_idx, run_idx in combos:
        key = (method, str(resolution), str(tile_size), str(image_idx), str(run_idx))
        if key in done:
            continue

        image_path = paths_by_res[resolution][image_idx]
        enhance_faces = method.endswith("gfpgan")
        timeout = timeout_general if "general" in method else timeout_other

        q = mp.Queue()
        p = mp.Process(target=_worker, args=(image_path, mode, tile_size, enhance_faces, q))
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            result = {"status": "timeout"}
            print(f"[runtime] TIMEOUT {method} res={resolution} tile={tile_size} img={image_idx} run={run_idx}")
        elif not q.empty():
            result = q.get()
        else:
            result = {"status": "crashed"}

        row = {
            "mode": method, "resolution": resolution, "tile_size": tile_size,
            "image_idx": image_idx, "run_idx": run_idx,
            "elapsed_seconds": result.get("elapsed", ""), "status": result.get("status"),
        }
        writer.writerow(row)
        f_out.flush()
        print(f"[runtime] {method:22s} res={resolution:4d} tile={tile_size:4d} img={image_idx} "
              f"run={run_idx} -> {row['status']} ({row['elapsed_seconds']})")

    f_out.close()
    summarize(out_path)


def summarize(raw_path=None):
    raw_path = raw_path or results_path("runtime_raw.csv")
    import pandas as pd
    df = pd.read_csv(raw_path)
    df["elapsed_seconds"] = pd.to_numeric(df["elapsed_seconds"], errors="coerce")

    status_counts = df.groupby(["mode", "resolution", "tile_size", "status"]).size().unstack(fill_value=0)

    ok = df[df["status"] == "ok"]
    summary = ok.groupby(["mode", "resolution", "tile_size"]).agg(
        n_ok=("elapsed_seconds", "count"),
        mean_seconds=("elapsed_seconds", "mean"),
        std_seconds=("elapsed_seconds", "std"),
    ).reset_index()
    summary = summary.merge(status_counts.reset_index(), on=["mode", "resolution", "tile_size"], how="outer")

    out_path = results_path("runtime_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"[runtime] wrote {out_path}")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-general", type=int, default=240,
                         help="seconds before killing a general_x4plus* config (slowest mode)")
    parser.add_argument("--timeout-other", type=int, default=90)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if args.summarize_only:
        summarize()
    else:
        run(args.n_images, args.repeats, args.seed, args.timeout_general, args.timeout_other)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
