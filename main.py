from utils.terminal_ui import print_header, print_step, suppress_warnings, clear_screen
suppress_warnings()

import os
import torch
from PIL import Image
import inquirer 

from utils.downloader import download_assets
from utils import config
from enhancement.enhancer import enhance_image
from captioning.generator import CaptionGenerator
from translation.translator import translate_caption
from utils.output_handler import send_to_discord, display_offline_report

def get_device():
    """Checks for the best available hardware and returns the device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_output_mode_from_terminal():
    """Asks the user to choose an output mode using a terminal menu."""
    questions = [
        inquirer.List('mode',
                      message="Choose output mode",
                      choices=['Online (Post to Discord)', 'Offline (Generate local report)'],
                      ),
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        print("No selection made. Exiting.")
        exit()
    return 'online' if 'Online' in answers['mode'] else 'offline'

def run_pipeline():
    clear_screen()
    
    print_header()
    output_mode = get_output_mode_from_terminal()

    print_step(0, "Setting up assets...")
    download_assets()

    print_step(1, "Enhancing image...")
    enhanced_image = enhance_image(config.IMAGE_PATH)
    base, ext = os.path.splitext(config.IMAGE_PATH)
    enhanced_image_path = f"{base}_enhanced{ext}"
    enhanced_image.save(enhanced_image_path)

    print_step(2, "Generating image caption...")
    device = get_device()
    caption_model = CaptionGenerator(device)
    english_caption = caption_model.generate_caption(enhanced_image)

    print_step(3, "Translating captions...")
    multilingual_captions = translate_caption(
        english_caption, 
        target_languages=config.TARGET_LANGUAGES
    )

    print_step(4, f"Finalizing output (Mode: {output_mode})...")
    if output_mode == 'online':
        send_to_discord(
            config.DISCORD_WEBHOOK_URL,
            config.DISCORD_SERVER_INVITE,
            enhanced_image_path,
            english_caption,
            multilingual_captions
        )
    else:
        display_offline_report(
            enhanced_image_path,
            english_caption,
            multilingual_captions
        )

if __name__ == '__main__':
    run_pipeline()
