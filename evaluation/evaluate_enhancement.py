"""
Evaluates C4PS's image-enhancement quality against known ground truth.

Methodology: start from a set of natural high-quality (HQ) images, apply a
controlled synthetic degradation (bicubic downsample, i.e. the classical SR
degradation model) to produce a low-quality (LQ) input whose "correct"
super-resolved output is exactly the original HQ image. This is the standard
way SR literature gets ground truth (real images have no known noise-free HQ
counterpart otherwise) -- it's what makes PSNR/SSIM/LPIPS against the HQ
image meaningful, unlike scoring an arbitrary enhanced social-media photo
against nothing.

Compares, at matching scale factors:
  - bicubic upsampling (naive baseline)
  - C4PS enhancement modes: fast (x2), general/x4plus, sharp_anime (x4),
    auto_vehicle (x4)

Usage:
    python -m evaluation.evaluate_enhancement --n-samples 20 --hq-size 512
"""
import argparse
import csv
import os
import time
import traceback

import numpy as np
from PIL import Image

from . import common  # noqa: E402
from .common import get_device, results_path, set_all_seeds, free_gpu_memory
from . import datasets
from . import metrics

X4_METHODS = {
    "general_x4plus": "general",
    "sharp_anime_x4": "sharp_anime",
    "auto_vehicle_x4": "auto_vehicle",
}


def degrade(hq_image: Image.Image, scale: int) -> Image.Image:
    """Bicubic downsample by `scale` -- the standard classical SR degradation."""
    w, h = hq_image.size
    lq = hq_image.resize((w // scale, h // scale), Image.BICUBIC)
    return lq


def bicubic_upsample(lq_image: Image.Image, scale: int) -> Image.Image:
    w, h = lq_image.size
    return lq_image.resize((w * scale, h * scale), Image.BICUBIC)


def to_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def run(n_samples, hq_size, seed, tile_size, device_for_lpips):
    set_all_seeds(seed)
    from enhancement.enhancer import enhance_image

    samples = datasets.load_flickr8k(n_samples=n_samples, split="train", seed=seed + 999)

    out_path = results_path("enhancement_raw.csv")
    fieldnames = ["image_id", "method", "scale", "psnr", "ssim", "lpips", "elapsed_seconds", "status"]
    f_out = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    tmp_lq_path = results_path("_tmp_enh_lq.png")

    for i, sample in enumerate(samples):
        image_id = sample["image_id"]
        print(f"[enhancement] ({i+1}/{len(samples)}) {image_id}")

        # Make HQ a fixed, scale-friendly size (divisible by 4) via center-crop-resize.
        hq = sample["image"].convert("RGB")
        w, h = hq.size
        side = min(w, h)
        hq = hq.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side))
        hq = hq.resize((hq_size, hq_size), Image.BICUBIC)
        hq_arr = to_array(hq)

        for scale, methods in [(4, {"bicubic_4x": None, **X4_METHODS}), (2, {"bicubic_2x": None, "fast_x2": "fast"})]:
            lq = degrade(hq, scale)
            lq.save(tmp_lq_path)

            for method, mode in methods.items():
                row = {"image_id": image_id, "method": method, "scale": scale}
                try:
                    t0 = time.time()
                    if mode is None:
                        out_img = bicubic_upsample(lq, scale)
                    else:
                        out_img = enhance_image(tmp_lq_path, mode=mode, tile_size=tile_size, enhance_faces=False)
                    elapsed = time.time() - t0

                    out_arr = to_array(out_img)
                    if out_arr.shape[:2] != hq_arr.shape[:2]:
                        out_img_resized = Image.fromarray(out_arr).resize((hq_size, hq_size), Image.BICUBIC)
                        out_arr = to_array(out_img_resized)

                    psnr_val, ssim_val = metrics.psnr_ssim(hq_arr, out_arr)
                    lpips_val = metrics.lpips_distance(hq_arr, out_arr, device=device_for_lpips)

                    row.update(psnr=psnr_val, ssim=ssim_val, lpips=lpips_val,
                               elapsed_seconds=elapsed, status="ok")
                    print(f"    {method}: PSNR={psnr_val:.2f} SSIM={ssim_val:.4f} LPIPS={lpips_val:.4f} ({elapsed:.1f}s)")
                    free_gpu_memory()

                except Exception as e:
                    print(f"[enhancement] ERROR {method} on {image_id}: {e}")
                    traceback.print_exc()
                    row.update(psnr="", ssim="", lpips="", elapsed_seconds="", status=f"error: {e}")

                writer.writerow(row)
                f_out.flush()

    f_out.close()
    if os.path.exists(tmp_lq_path):
        os.remove(tmp_lq_path)

    summarize(out_path)


def summarize(raw_path=None):
    raw_path = raw_path or results_path("enhancement_raw.csv")
    import pandas as pd
    df = pd.read_csv(raw_path)
    df_ok = df[df["status"] == "ok"].copy()
    for col in ["psnr", "ssim", "lpips", "elapsed_seconds"]:
        df_ok[col] = pd.to_numeric(df_ok[col], errors="coerce")

    summary = df_ok.groupby(["method", "scale"]).agg(
        n=("psnr", "count"),
        psnr_mean=("psnr", "mean"), psnr_std=("psnr", "std"),
        ssim_mean=("ssim", "mean"), ssim_std=("ssim", "std"),
        lpips_mean=("lpips", "mean"), lpips_std=("lpips", "std"),
        elapsed_mean=("elapsed_seconds", "mean"),
    ).reset_index()

    out_path = results_path("enhancement_metrics.csv")
    summary.to_csv(out_path, index=False)
    print(f"[enhancement] wrote {out_path}")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--hq-size", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    device = get_device()
    lpips_device = "cuda" if device.type == "cuda" else "cpu"

    if args.summarize_only:
        summarize()
    else:
        run(args.n_samples, args.hq_size, args.seed, args.tile_size, lpips_device)


if __name__ == "__main__":
    main()
