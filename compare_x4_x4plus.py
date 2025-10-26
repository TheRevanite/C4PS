import os
import sys
import time
import numpy as np
from PIL import Image

# Monkey-patch: make torchvision.transforms.functional_tensor available
try:
    from torchvision.transforms.v2 import functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except Exception:
    # fall back if torchvision version doesn't have v2.functional
    try:
        import torchvision.transforms.functional as F2
        sys.modules['torchvision.transforms.functional_tensor'] = F2
    except Exception:
        pass

def load_image_np(path):
    img = Image.open(path).convert('RGB')
    return np.array(img)

def save_image_np(arr, path):
    Image.fromarray(arr).save(path)

def enhance_x4_direct(weight_path, input_np):
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
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
    img = input_np.astype('float32') / 255.0
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        t0 = time.time()
        out = model(img)
        elapsed = time.time() - t0
    out_np = out.squeeze(0).permute(1,2,0).cpu().numpy()
    out_np = (out_np * 255.0).clip(0,255).astype('uint8')
    return out_np, elapsed

def enhance_x4plus(weight_path, input_np, tile_size=400):
    from enhancement.model import EnhancementModel
    model = EnhancementModel(model_path=weight_path, tile_size=tile_size)
    t0 = time.time()
    out = model.upscale(input_np)
    elapsed = time.time() - t0
    return out.astype('uint8'), elapsed

def compute_metrics(ref_np, out_np):
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
    except Exception:
        return None, None
    ref = ref_np.astype('uint8')
    out = out_np.astype('uint8')
    # resize ref if different
    if ref.shape != out.shape:
        ref = np.array(Image.fromarray(ref).resize((out.shape[1], out.shape[0]), Image.BICUBIC))
    ps = psnr(ref, out, data_range=255)
    ss = ssim(ref, out, data_range=255, channel_axis=-1)
    return ps, ss

def make_report(input_path, reference_path, weight_x4, weight_x4plus, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    inp = load_image_np(input_path)
    ref = load_image_np(reference_path)

    print('Running x4 direct...')
    out_x4, t_x4 = enhance_x4_direct(weight_x4, inp)
    p_x4, s_x4 = compute_metrics(ref, out_x4)
    out_x4_path = os.path.join(out_dir, 'out_x4.png')
    save_image_np(out_x4, out_x4_path)

    print('Running x4plus...')
    out_x4p, t_x4p = enhance_x4plus(weight_x4plus, inp, tile_size=400)
    p_x4p, s_x4p = compute_metrics(ref, out_x4p)
    out_x4p_path = os.path.join(out_dir, 'out_x4plus.png')
    save_image_np(out_x4p, out_x4p_path)

    # save input and reference copies
    in_path = os.path.join(out_dir, 'input.png')
    ref_path = os.path.join(out_dir, 'reference.png')
    save_image_np(inp.astype('uint8'), in_path)
    save_image_np(ref.astype('uint8'), ref_path)

    # build HTML
    html = f"""
    <html>
      <head><title>Enhancement comparison</title></head>
      <body>
        <h2>Enhancement comparison: x4 vs x4plus</h2>
        <table border="1" cellpadding="8">
          <tr>
            <th>Input</th>
            <th>x4 (direct)</th>
            <th>x4plus</th>
            <th>Reference</th>
          </tr>
          <tr>
            <td><img src="{os.path.basename(in_path)}" style="max-width:300px"></td>
            <td><img src="{os.path.basename(out_x4_path)}" style="max-width:300px"><br>time: {t_x4:.2f}s<br>PSNR: {p_x4 if p_x4 is not None else 'n/a'}<br>SSIM: {s_x4 if s_x4 is not None else 'n/a'}</td>
            <td><img src="{os.path.basename(out_x4p_path)}" style="max-width:300px"><br>time: {t_x4p:.2f}s<br>PSNR: {p_x4p if p_x4p is not None else 'n/a'}<br>SSIM: {s_x4p if s_x4p is not None else 'n/a'}</td>
            <td><img src="{os.path.basename(ref_path)}" style="max-width:300px"></td>
          </tr>
        </table>
      </body>
    </html>
    """

    report_path = os.path.join(out_dir, 'report.html')
    with open(report_path, 'w') as f:
        f.write(html)

    print('Report written to', report_path)
    return report_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare x4 vs x4plus weights')
    parser.add_argument('input', help='input image path')
    parser.add_argument('reference', help='reference image path')
    parser.add_argument('--x4', default='weights/RealESRGAN_x4.pth', help='x4 weight path')
    parser.add_argument('--x4plus', default='weights/RealESRGAN_x4plus.pth', help='x4plus weight path')
    parser.add_argument('--out', default='outputs', help='output folder')
    args = parser.parse_args()
    make_report(args.input, args.reference, args.x4, args.x4plus, out_dir=args.out)
