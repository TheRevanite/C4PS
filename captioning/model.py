from transformers import AutoProcessor, AutoModelForCausalLM
from utils import config

def create_transformer_model(device):
    """
    Loads a pre-trained GIT (Generative Image-to-Text) model from Hugging Face,
    saving it to the local project directory specified in the config file.

    Args:
        device (torch.device): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing the model and its processor.
    """
    
    model_name = "microsoft/git-base-coco"
    
    processor = AutoProcessor.from_pretrained(
        model_name, 
        cache_dir=config.CAPTIONING_MODEL_PATH
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=config.CAPTIONING_MODEL_PATH
    ).to(device)

    return model, processor
