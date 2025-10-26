import sys
import os
try:
    import psutil
except Exception:
    psutil = None
from torchvision.transforms.v2 import functional as F

# --- MONKEY-PATCH FIX ---
sys.modules['torchvision.transforms.functional_tensor'] = F
# --- END OF FIX ---

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from utils import config

# <-- MODIFIED IMPORT BLOCK ---
GFPGAN_AVAILABLE = False # Assume it's not available initially
try:
    print("[DEBUG] Attempting to import GFPGAN...")
    from gfpgan import GFPGANer
    # from gfpgan.utils import GFPGANer_post_process # <-- REMOVED THIS LINE
    import facexlib
    print("[DEBUG] GFPGAN imported successfully!")
    GFPGAN_AVAILABLE = True
except ImportError as e:
    print(f"[DEBUG] Failed to import GFPGAN. Error: {e}")
    GFPGAN_AVAILABLE = False
# --- END OF MODIFIED BLOCK ---

# Reduce thread usage to lower memory pressure
os.environ.setdefault('OMP_NUM_THREADS', '1')
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

def get_device():
    """Checks for the best available hardware and returns the device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            if torch.backends.mps.is_built():
                return torch.device("mps")
            else:
                print("[WARNING] MPS backend is available but not built. Falling back to CPU.")
        except AttributeError:
            return torch.device("mps")
    return torch.device("cpu")

class EnhancementModel:
    """
    This class encapsulates the Real-ESRGAN model for image enhancement.
    This is ONLY used for 'checkpoint' models.
    """
    def __init__(self, model_params, outscale, model_path, tile_size=800, face_enhance=False):
        self.device = get_device()
        self.tile_size = tile_size
        self.outscale = outscale
        self.do_face_enhance = face_enhance # Store the choice

        # Build model from params, RealESRGANer will load weights from model_path
        if model_params is None:
            raise ValueError("Must provide 'model_params' for EnhancementModel")

        print(f"[INFO] Initializing model from checkpoint: {model_path}")
        model = RRDBNet(**model_params)

        use_half = self.device.type == 'cuda'
        if self.device.type == 'mps':
            print("[INFO] Using MPS (Apple Silicon) device. Half precision is not supported on MPS; using full precision.")

        # --- FACE ENHANCER INITIALIZATION ---
        self.face_enhancer = None # Store the GFPGANer instance
        if self.do_face_enhance:
            if not GFPGAN_AVAILABLE:
                print("[WARNING] `gfpgan` package check failed (import error). Face enhancement disabled.")
                self.do_face_enhance = False
            else:
                gfpgan_path = os.path.join('weights', 'GFPGANv1.4.pth')
                if not os.path.exists(gfpgan_path):
                    print(f"[WARNING] GFPGAN model not found at {gfpgan_path}. Face enhancement disabled.")
                    print("[INFO] Download GFPGANv1.4.pth and place it in the 'weights' folder.")
                    self.do_face_enhance = False
                else:
                    try:
                        print("[INFO] Initializing GFPGAN for face enhancement...")
                        # Set root_dir for clean cache
                        facexlib.utils.load_file_from_url.root_dir = os.path.join(
                            os.path.expanduser('~'), '.cache/gfpgan'
                        )

                        # <-- UPDATED GFPGANer CALL (removed post_process)
                        self.face_enhancer = GFPGANer(
                            model_path=gfpgan_path,
                            upscale=outscale,
                            arch='clean',
                            channel_multiplier=2,
                            bg_upsampler=None,
                            device=self.device
                            # post_process argument removed
                        )
                    except Exception as e:
                        print(f"[ERROR] Failed to initialize GFPGAN: {e}. Face enhancement disabled.")
                        self.do_face_enhance = False
        # --- END OF FACE LOGIC ---

        # --- RAM check removed ---
        capped_tile = tile_size
        print(f"[ENHANCEMENT] Requested tile size: {tile_size} (RAM safety check disabled)")

        # Create RealESRGANer WITHOUT face_enhancer argument
        self.upsampler = RealESRGANer(
            scale=self.outscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=capped_tile,
            tile_pad=10,
            pre_pad=0,
            half=use_half,
            gpu_id=0 if self.device.type == 'cuda' else None,
        )
        self._capped_tile = capped_tile

    def upscale(self, image_array, reference_image=None):
        """
        Performs the enhancement on a numpy image array, optionally with face enhancement.
        """
        import time
        try:
            from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
            skimage_available = True
        except ImportError:
            skimage_available = False

        print(f"[ENHANCEMENT] Input image shape: {getattr(image_array, 'shape', None)}")
        if max(getattr(image_array, 'shape', [0,0,0])[:2]) > 2000:
            print("[WARNING] Large image detected. Consider resizing for faster enhancement or increasing RAM/tile size.")

        start_total = time.time()
        output = None # Initialize output

        # --- STEP 1: RealESRGAN Enhancement ---
        start_realesrgan = time.time()
        attempts = 0
        max_attempts = 3
        current_tile = getattr(self, '_capped_tile', self.tile_size)

        while attempts < max_attempts:
            try:
                # Run RealESRGAN enhancement
                output, _ = self.upsampler.enhance(
                    image_array,
                    outscale=self.outscale
                )
                break
            except (RuntimeError, MemoryError) as e:
                attempts += 1
                print(f"[WARNING] RealESRGAN failed on attempt {attempts} with tile={current_tile}: {e}")

                # reduce tile and recreate upsampler
                current_tile = max(128, current_tile // 2)
                try:
                    existing_model = self.upsampler.net_g if hasattr(self.upsampler, 'net_g') else None
                    existing_path = self.upsampler.model_path if hasattr(self.upsampler, 'model_path') else None

                    self.upsampler = RealESRGANer(
                        scale=self.outscale,
                        model_path=existing_path,
                        dni_weight=None,
                        model=existing_model,
                        tile=current_tile,
                        tile_pad=10,
                        pre_pad=0,
                        half=False if self.device.type == 'mps' else (self.device.type == 'cuda'),
                        gpu_id=0 if self.device.type == 'cuda' else None,
                    )
                    print(f"[INFO] Retrying RealESRGAN with smaller tile: {current_tile}")
                    continue
                except Exception as e2:
                    print(f"[ERROR] Failed to recreate RealESRGAN upsampler with smaller tile: {e2}")
                    return image_array # Return original on failure
        else:
            print("[ERROR] RealESRGAN enhancement failed after retries. Returning original image.")
            return image_array

        elapsed_realesrgan = time.time() - start_realesrgan
        print(f"[BENCHMARK] RealESRGAN took {elapsed_realesrgan:.3f} seconds.")

        # --- STEP 2: GFPGAN Face Enhancement (if enabled) ---
        if self.do_face_enhance and self.face_enhancer is not None:
            start_gfpgan = time.time()
            try:
                _, _, output = self.face_enhancer.enhance(
                    output,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )
                elapsed_gfpgan = time.time() - start_gfpgan
                print(f"[BENCHMARK] GFPGAN face enhancement took {elapsed_gfpgan:.3f} seconds.")
            except Exception as e:
                print(f"[ERROR] GFPGAN enhancement failed: {e}. Returning RealESRGAN result.")

        elapsed_total = time.time() - start_total
        print(f"[BENCHMARK] Total enhancement took {elapsed_total:.3f} seconds.")

        # --- Quality Metrics ---
        if reference_image is not None and skimage_available:
            try:
                import numpy as np
                ref = reference_image.astype(np.uint8)
                out = output.astype(np.uint8)
                if ref.shape == out.shape:
                    psnr_val = psnr(ref, out, data_range=255)
                    ssim_val = ssim(ref, out, data_range=255, channel_axis=-1)
                    print(f"[BENCHMARK] PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
                else:
                    print("[BENCHMARK] Reference and output image shapes do not match; skipping quality metrics.")
            except Exception as e:
                print(f"[BENCHMARK] Error computing quality metrics: {e}")
        elif reference_image is not None:
            print("[BENCHMARK] skimage not available; install scikit-image for quality metrics.")

        return output