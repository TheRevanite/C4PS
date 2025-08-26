import sys
from torchvision.transforms.v2 import functional as F

# --- MONKEY-PATCH FIX ---
sys.modules['torchvision.transforms.functional_tensor'] = F
# --- END OF FIX ---

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from utils import config

def get_device():
    """Checks for the best available hardware and returns the device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class EnhancementModel:
    """
    This class encapsulates the Real-ESRGAN model for image enhancement.
    """
    def __init__(self, model_path=config.ENHANCER_WEIGHTS_PATH):
        self.device = get_device()
        
        model = RRDBNet(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=64, 
            num_block=23, 
            num_grow_ch=32, 
            scale=4
        )
        
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True if self.device.type != 'cpu' else False,
            gpu_id=0 if self.device.type == 'cuda' else None
        )

    def upscale(self, image_array):
        """
        Performs the enhancement on a numpy image array.
        """
        try:
            output, _ = self.upsampler.enhance(image_array, outscale=4)
            return output
        except Exception as e:
            print(f"[ERROR] Error during upscaling: {e}. Returning original.")
            return image_array