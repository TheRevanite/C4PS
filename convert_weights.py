import torch
import os

# Files to convert: (input_filename, output_filename)
files_to_convert = [
    ('RealESRGAN_x2.pth', 'RealESRGAN_x2_chk.pth'),
    ('RealESRGAN_x4.pth', 'RealESRGAN_x4_chk.pth'),
]

# The key the 'realesrgan' library expects
# (based on 'params_ema' from your sweep script)
KEY = 'params_ema' 

print("Starting weight conversion...")

for in_file, out_file in files_to_convert:
    in_path = os.path.join('weights', in_file)
    out_path = os.path.join('weights', out_file)

    if not os.path.exists(in_path):
        print(f"[SKIP] Input file not found: {in_path}")
        continue
    
    if os.path.exists(out_path):
        print(f"[SKIP] Output file already exists: {out_path}")
        continue

    try:
        print(f"Converting '{in_file}' -> '{out_file}'...")
        
        # Load the raw weights
        state_dict = torch.load(in_path, map_location='cpu')
        
        # Create the new checkpoint dictionary
        checkpoint = {KEY: state_dict}
        
        # Save the new checkpoint file
        torch.save(checkpoint, out_path)
        
        print(f"[SUCCESS] Created new checkpoint file: {out_path}")

    except Exception as e:
        print(f"[ERROR] Failed to convert {in_file}: {e}")

print("Conversion complete.")