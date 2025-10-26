#!/usr/bin/env python3
import os
import sys
import csv
import time
import numpy as np
from PIL import Image

# Monkey-patch like other modules
try:
    from torchvision.transforms.v2 import functional as F
    import sys as _sys
    _sys.modules['torchvision.transforms.functional_tensor'] = F
except Exception:
    try:
        import torchvision.transforms.functional as F2
        import sys as _sys
        _sys.modules['torchvision.transforms.functional_tensor'] = F2
    except Exception:
        pass

def read_sweep_results(path='sweep_results.csv'):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def pick_fast(rows):
    # Choose fastest weight among those whose PSNR is within 1.0 dB of best
    valid = [r for r in rows if r.get('status') == 'ok' and r.get('psnr')]
    if not valid:
        # fallback to any ok
        valid = [r for r in rows if r.get('status') == 'ok']
    if not valid:
        return None
    psnrs = [float(r['psnr']) for r in valid if r.get('psnr')]
    if psnrs:
        best = max(psnrs)
        threshold = best - 1.0
        candidates = [r for r in valid if r.get('psnr') and float(r['psnr']) >= threshold]
        if not candidates:
            candidates = valid
    else:
        candidates = valid
    # pick minimal elapsed
    candidates_with_elapsed = [r for r in candidates if r.get('elapsed')]
    if candidates_with_elapsed:
        best_row = min(candidates_with_elapsed, key=lambda r: float(r['elapsed']))
    else:
        best_row = candidates[0]
    return best_row

def pick_accurate(rows, tile_choice):
    # Choose highest PSNR among given tile_size
    candidates = [r for r in rows if r.get('status') == 'ok' and r.get('tile_size') == str(tile_choice) and r.get('psnr')]
    if not candidates:
        candidates = [r for r in rows if r.get('status') == 'ok' and r.get('tile_size') == str(tile_choice)]
    if not candidates:
        # fallback to any ok
        candidates = [r for r in rows if r.get('status') == 'ok']
    if not candidates:
        return None
    # pick highest psnr, tie-breaker lowest elapsed
    candidates = sorted(candidates, key=lambda r: (-(float(r['psnr']) if r.get('psnr') else -9999), float(r['elapsed']) if r.get('elapsed') else 1e9))
    return candidates[0]

def load_image_np(path):
    img = Image.open(path).convert('RGB')
    return np.array(img)

def run_enhancement(selected, input_path, reference_path, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    weight = selected['weight']
    tile = int(selected['tile_size'])
    weight_path = os.path.join('weights', weight) if not os.path.isabs(weight) else weight
    print(f"Selected weight: {weight_path}, tile: {tile}")
    img_np = load_image_np(input_path)
    ref_np = load_image_np(reference_path) if reference_path else None

    # Decide model type
    model_type = None
    if 'x4plus' in weight.lower():
        model_type = 'x4plus'
    else:
        # parse x2/x4/x8 from filename
        if 'x2' in weight.lower():
            model_type = 2
        elif 'x4' in weight.lower():
            model_type = 4
        elif 'x8' in weight.lower():
            model_type = 8
        else:
            model_type = 4

    if model_type == 'x4plus':
        # use EnhancementModel
        from enhancement.model import EnhancementModel
        enhancer = EnhancementModel(model_path=weight_path, tile_size=tile)
        t0 = time.time()
        out = enhancer.upscale(img_np)
        elapsed = time.time() - t0
        out_np = out.astype('uint8')
    else:
        # Attempt direct RRDBNet inference; if basicsr is not installed, fall back to EnhancementModel
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=model_type)
            loadnet = torch.load(weight_path, map_location=lambda storage, loc: storage)
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
            img = img_np.astype('float32') / 255.0
            img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
            with torch.no_grad():
                t0 = time.time()
                out = model(img)
                elapsed = time.time() - t0
            out_np = out.squeeze(0).permute(1,2,0).cpu().numpy()
            out_np = (out_np * 255.0).clip(0,255).astype('uint8')
        except ModuleNotFoundError as e:
            # fallback to EnhancementModel if basicsr/realesrgan not installed
            print(f"Module not found: {e}. Trying fallback to EnhancementModel...")
            try:
                from enhancement.model import EnhancementModel
                enhancer = EnhancementModel(model_path=weight_path, tile_size=tile)
                t0 = time.time()
                out = enhancer.upscale(img_np)
                elapsed = time.time() - t0
                out_np = out.astype('uint8')
            except Exception as e2:
                print(f"Fallback to EnhancementModel failed: {e2}")
                print("Please install the requirements: pip install basicsr realesrgan scikit-image or run this inside your .venv where they are installed.")
                raise

    out_path = os.path.join(out_dir, 'selected_output.png')
    Image.fromarray(out_np).save(out_path)
    print(f"Saved output to {out_path} (elapsed {elapsed:.2f}s)")

    # compute metrics
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
        skimage_available = True
    except Exception:
        skimage_available = False
    if ref_np is not None and skimage_available:
        ref = ref_np.astype('uint8')
        out = out_np.astype('uint8')
        if ref.shape != out.shape:
            ref = np.array(Image.fromarray(ref).resize((out.shape[1], out.shape[0]), Image.BICUBIC))
        p = psnr(ref, out, data_range=255)
        s = ssim(ref, out, data_range=255, channel_axis=-1)
        print(f"PSNR: {p:.2f} dB, SSIM: {s:.4f}")
    else:
        if ref_np is None:
            print("No reference provided; skipping quality metrics.")
        else:
            print("scikit-image not available; install scikit-image to compute PSNR/SSIM.")

def main():
    if len(sys.argv) < 3:
        print("Usage: choose_and_run.py <input_image> <reference_image>")
        sys.exit(1)
    input_path = sys.argv[1]
    ref_path = sys.argv[2]
    rows = read_sweep_results('sweep_results.csv')
    if not rows:
        print("No sweep_results.csv found — run sweep_enhancement.py first to collect data.")
        sys.exit(1)

    print("Choose mode:\n 1) Short & fast\n 2) Long & more accurate")
    choice = input("Enter 1 or 2: ").strip()
    if choice == '1':
        selected = pick_fast(rows)
        if not selected:
            print("No suitable weight found.")
            sys.exit(1)
        print(f"Auto selected: {selected['weight']} (tile {selected['tile_size']})")
        run_enhancement(selected, input_path, ref_path)
    elif choice == '2':
        ts = input("Choose tile size: 400 (faster) or 800 (better): ").strip()
        if ts not in ('400','800'):
            print("Invalid tile size; defaulting to 400")
            ts = '400'
        selected = pick_accurate(rows, int(ts))
        if not selected:
            print("No suitable weight found for that tile size.")
            sys.exit(1)
        print(f"Selected for accuracy: {selected['weight']} (tile {selected['tile_size']})")
        run_enhancement(selected, input_path, ref_path)
    else:
        print("Invalid choice")

if __name__ == '__main__':
    main()
