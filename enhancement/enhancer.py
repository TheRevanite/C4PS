import numpy as np
from PIL import Image
import os
import torch

# Import from our own model file
from .model import EnhancementModel

# Define model parameters
MODELS_CONFIG = {
    'fast': {
        'weight_file': 'RealESRGAN_x2_chk.pth', # (Uses converted file)
        'load_mode': 'checkpoint',
        'model_params': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 23,
            'num_grow_ch': 32,
            'scale': 2
        },
        'outscale': 2
    },
    'general': {
        'weight_file': 'RealESRGAN_x4plus.pth', # (This file is already a checkpoint)
        'load_mode': 'checkpoint',
        'model_params': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 23,
            'num_grow_ch': 32,
            'scale': 4
        },
        'outscale': 4
    },
    'sharp_anime': {
        # Real-ESRGAN's official anime-optimized checkpoint (6-block RRDBNet,
        # trained for flatter/sharper illustrated content). Previously this
        # mode had no entry here, so main.py's "Sharp (Vehicles/Anime)" menu
        # option silently fell through the MODELS_CONFIG.get(mode, general)
        # default and ran the general x4plus model instead -- i.e. it was
        # never actually distinct from 'general'.
        'weight_file': 'RealESRGAN_x4plus_anime_6B.pth',
        'load_mode': 'checkpoint',
        'model_params': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 6,
            'num_grow_ch': 32,
            'scale': 4
        },
        'outscale': 4
    },
    # 'auto_vehicle' has no distinct trained checkpoint of its own -- it
    # intentionally reuses the general x4plus model until/unless a
    # vehicle-specific model is trained. Left as an explicit alias (rather
    # than relying on MODELS_CONFIG.get(mode, general)'s silent fallback)
    # so this is a documented decision, not indistinguishable from a bug.
}
MODELS_CONFIG['auto_vehicle'] = MODELS_CONFIG['general']


def enhance_image(image_path, mode, tile_size, face_enhance=False, enhance_faces=None):
    """
    Enhances and upscales an image using the Real-ESRGAN model.
    """
    # Prefer the newer `enhance_faces` keyword when provided by callers.
    if enhance_faces is not None:
        face_enhance = enhance_faces

    try:
        img_array = np.array(Image.open(image_path).convert('RGB'))
        
        # Select config, default to 'general' if mode is invalid
        config = MODELS_CONFIG.get(mode, MODELS_CONFIG['general'])
        
        weight_file = config['weight_file']
        weight_path = weight_file if os.path.isabs(weight_file) else os.path.join('weights', weight_file)

        if not os.path.exists(weight_path):
            print(f"[ERROR] Weight file not found: {weight_path}")
            print("[INFO] Did you run 'python3 convert_weights.py'?")
            return Image.open(image_path)

        # This is the single, clean path for ALL models now.
        print(f"[INFO] Using CHECKPOINT model path (RealESRGANer): {weight_path}")
        
        # <-- THIS IS THE FIX: Pass 'face_enhance' to constructor
        enhancer = EnhancementModel(
            model_params=config['model_params'],
            outscale=config['outscale'],
            model_path=weight_path,
            tile_size=tile_size,
            face_enhance=face_enhance 
        )
        
        # <-- THIS IS THE FIX: 'face_enhance' argument removed from here
        output_array = enhancer.upscale(img_array)
        return Image.fromarray(output_array)

    except Exception as e:
        print(f"[ERROR] Could not read or enhance image: {e}. Returning original.")
        import traceback
        traceback.print_exc()
        return Image.open(image_path)