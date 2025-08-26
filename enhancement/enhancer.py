import numpy as np
from PIL import Image
from .model import EnhancementModel

enhancement_model = EnhancementModel()

def enhance_image(image_path):
    """
    Enhances and upscales an image using the loaded Real-ESRGAN model.
    
    Args:
        image_path (str): Path to the input image.

    Returns:
        PIL.Image.Image: The enhanced image.
    """
    try:
        img = np.array(Image.open(image_path).convert('RGB'))
        output_array = enhancement_model.upscale(img)
        return Image.fromarray(output_array)
    
    except Exception as e:
        print(f"[ERROR] Could not read or enhance image: {e}. Returning a placeholder.")
        return Image.open(image_path)

