import torch
from .model import create_transformer_model

class CaptionGenerator:
    """
    Orchestrates the captioning process using the pre-trained Transformer model.
    """
    def __init__(self, device):
        self.device = device
        self.model, self.processor = create_transformer_model(device)
        self.model.eval()

    def generate_caption(self, image):
        """
        Generates a caption for a given PIL image using the Transformer model.
        """
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=50)
            
        caption = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return caption