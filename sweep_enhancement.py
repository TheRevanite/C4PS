"""
Benchmark multiple Real-ESRGAN weights and tile sizes for speed and quality.

Usage:
    python sweep_enhancement.py input.png reference.png

Edit the WEIGHTS and TILE_SIZES lists to try different models/weights and tile sizes.

Download official weights from:
- https://github.com/xinntao/Real-ESRGAN#model-zoo
  (e.g., RealESRGAN_x4plus.pth, RealESRGAN_x4plus_anime_6B.pth, etc.)
Place them in your weights/ directory and update the paths below.
"""

import sys
import os
import csv
import time
import multiprocessing as mp
import numpy as np
from PIL import Image
from enhancement.model import EnhancementModel, get_device
from realesrgan import RealESRGANer

# List of weights to try (add your own paths)


WEIGHTS = [
    ('weights/RealESRGAN_x2.pth', 2),
    ('weights/RealESRGAN_x4.pth', 4),
    ('weights/RealESRGAN_x4plus.pth', 'x4plus'),
    ('weights/RealESRGAN_x8.pth', 8),
]

# List of tile sizes to try
TILE_SIZES = [400, 800, 1200]

def load_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img)


def worker_process(weight, model_type, tile, input_path, ref_path, q):
    """Worker that runs one benchmark combo and returns a dict via queue."""
    try:
        input_img = load_image(input_path)
        ref_img = load_image(ref_path) if ref_path else None
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import torch
        if model_type == 'x4plus':
            # Use EnhancementModel (which wraps RealESRGANer)
            enhancer = EnhancementModel(model_path=weight, tile_size=tile)
            start = time.time()
            out = enhancer.upsampler.enhance(input_img, outscale=4)
            # RealESRGANer.enhance returns (output, None) typically
            output = out[0]
            elapsed = time.time() - start
            out_img = output.astype('uint8')
        else:
            device = get_device()
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=model_type)
            loadnet = torch.load(weight, map_location=lambda storage, loc: storage)
            if isinstance(loadnet, dict):
                if 'params_ema' in loadnet:
                    state_dict = loadnet['params_ema']
                elif 'params' in loadnet:
                    state_dict = loadnet['params']
                elif all(isinstance(v, torch.Tensor) for v in loadnet.values()):
                    state_dict = loadnet
                else:
                    state_dict = list(loadnet.values())[0]
            else:
                state_dict = loadnet
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            model = model.to(device)
            img = input_img.astype('float32') / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                s = time.time()
                output = model(img)
                elapsed = time.time() - s
            out_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_img = (out_img * 255.0).clip(0, 255).astype('uint8')

        # compute metrics
        psnr_val = None
        ssim_val = None
        try:
            from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
            skimage_available = True
        except Exception:
            skimage_available = False
        if ref_img is not None and skimage_available:
            import numpy as np
            from PIL import Image as PILImage
            ref = ref_img.astype(np.uint8)
            out = out_img.astype(np.uint8)
            if ref.shape != out.shape:
                ref = np.array(PILImage.fromarray(ref).resize((out.shape[1], out.shape[0]), PILImage.BICUBIC))
            psnr_val = psnr(ref, out, data_range=255)
            ssim_val = ssim(ref, out, data_range=255, channel_axis=-1)

        q.put({'status': 'ok', 'elapsed': elapsed, 'psnr': psnr_val, 'ssim': ssim_val})
    except Exception as e:
        q.put({'status': 'error', 'error': str(e)})

def main():
    # Ensure dummy.pth exists for bypassing RealESRGANer loader
    with open('dummy.pth', 'wb') as f:
        pass
    if len(sys.argv) != 3:
        print("Usage: python sweep_enhancement.py <input_image> <reference_image>")
        sys.exit(1)
    input_path, ref_path = sys.argv[1], sys.argv[2]
    input_img = load_image(input_path)
    ref_img = load_image(ref_path)
    # Results CSV (append/resume)
    results_file = 'sweep_results.csv'
    seen = set()
    if os.path.exists(results_file):
        with open(results_file, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                seen.add((r['weight'], r['tile_size']))

    for weight, model_type in WEIGHTS:
        if not os.path.exists(weight):
            print(f"[SKIP] Weight not found: {weight}")
            continue
        for tile in TILE_SIZES:
            key = (os.path.basename(weight), str(tile))
            if key in seen:
                print(f"[SKIP] Already done: {key}")
                continue
            print(f"\n=== Benchmarking: weight={os.path.basename(weight)}, tile_size={tile} ===")
            q = mp.Queue()
            p = mp.Process(target=worker_process, args=(weight, model_type, tile, input_path, ref_path, q))
            p.start()
            # timeout: give more time to x4plus combos
            timeout = 180 if model_type == 'x4plus' else 90
            p.join(timeout)
            if p.is_alive():
                print(f"[TIMEOUT] Combo timed out after {timeout}s; terminating process.")
                p.terminate()
                p.join()
                result = {'status': 'timeout'}
            else:
                if not q.empty():
                    result = q.get()
                else:
                    result = {'status': 'noresult'}

            # write to CSV
            row = {
                'weight': os.path.basename(weight),
                'tile_size': str(tile),
                'status': result.get('status'),
                'elapsed': result.get('elapsed'),
                'psnr': result.get('psnr'),
                'ssim': result.get('ssim')
            }
            write_header = not os.path.exists(results_file)
            with open(results_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['weight','tile_size','status','elapsed','psnr','ssim'])
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            seen.add(key)

if __name__ == "__main__":
    main()
