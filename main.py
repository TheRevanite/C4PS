import sys
from torchvision.transforms.v2 import functional as F
# --- MONKEY-PATCH FIX ---
# This patch must run BEFORE basicsr is imported by any other file
sys.modules['torchvision.transforms.functional_tensor'] = F
# --- END OF FIX ---

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
                      message="[?] Choose output mode",
                      choices=['Online (Post to Discord)', 'Offline (Generate local report)'],
                      ),
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        print("No selection made. Exiting.")
        exit()
    return 'online' if 'Online' in answers['mode'] else 'offline'

def get_enhancement_mode_from_terminal():
    """Asks the user to choose an enhancement mode based on benchmark data."""
    questions = [
        inquirer.List('mode',
                      message="[?] Choose enhancement quality",
                      choices=[
                          'Fast (x2)        | ~30 sec | Low VRAM,Faster and lower quality',
                          'General (x4plus) | ~150 sec | High VRAM, Slower and higher quality'
                      ],
                      ),
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        print("No selection made. Exiting.")
        exit()

    if 'Fast (x2)' in answers['mode']:
        return 'fast'

    return 'general'

def get_tile_size_from_terminal():
    """Asks the user to choose a tile size."""
    questions = [
        inquirer.List('tile_size',
                      message="[?] Choose processing tile size (800 is a safe default)",
                      choices=[
                          '400 (Lowest VRAM use)',
                          '800 (Recommended Default)',
                          '1200 (High VRAM use)',
                          '1400 (Very High VRAM use)'
                      ],
                      default='800 (Recommended Default)',
                      ),
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        print("No selection made. Exiting.")
        exit()

    # Extract just the number
    return int(answers['tile_size'].split(' ')[0])

def get_face_enhance_choice():
    """Asks the user if they want to enable face enhancement using a list."""
    questions = [
        inquirer.List('face_enhance',
                         message="[?] Attempt to restore faces (GFPGAN)?",
                         choices=['Yes', 'No'],
                         default='No'),
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        return False # Default to No if no selection
    return answers['face_enhance'] == 'Yes'

def run_pipeline():
    clear_screen()

    print_header()
    output_mode = get_output_mode_from_terminal()
    enhancement_mode = get_enhancement_mode_from_terminal()
    tile_size = get_tile_size_from_terminal()
    face_enhance = get_face_enhance_choice()
    if face_enhance:
        print("[INFO] Face enhancement (GFPGAN) enabled.")

    print_step(0, "Setting up assets...")
    download_assets()

    # Initialize models once
    device = get_device()
    caption_model = CaptionGenerator(device)

    # --- UPDATED HYBRID PIPELINE LOGIC ---

    # Determine captioning strategy based on mode and tile size
    caption_original_first = (enhancement_mode == 'fast' and tile_size <= 400)

    if caption_original_first:
        # --- Caption FIRST (for Fast mode with small tile) ---
        print_step(1, "Generating image caption (from original)...")
        try:
            image_to_caption = Image.open(config.IMAGE_PATH)
        except Exception as e:
            print(f"[ERROR] Could not open original image at {config.IMAGE_PATH}: {e}")
            return

        english_caption = caption_model.generate_caption(image_to_caption)
        print(f"[INFO] Generated caption: \"{english_caption}\"")

        print_step(2, f"Enhancing image ({enhancement_mode})...")
        enhanced_image = enhance_image(
            config.IMAGE_PATH,
            mode=enhancement_mode,
            tile_size=tile_size,
            face_enhance=face_enhance
        )
    else:
        # --- Enhance FIRST (for General mode OR Fast mode with large tile) ---
        print_step(1, f"Enhancing image ({enhancement_mode})...")
        enhanced_image = enhance_image(
            config.IMAGE_PATH,
            mode=enhancement_mode,
            tile_size=tile_size,
            face_enhance=face_enhance
        )

        print_step(2, "Generating image caption (from enhanced)...")
        # 'enhanced_image' is already a PIL object from the enhancer
        english_caption = caption_model.generate_caption(enhanced_image)
        print(f"[INFO] Generated caption: \"{english_caption}\"")

    # --- END OF HYBRID LOGIC ---


    # --- COMMON STEPS ---

    # Step 3: Save the final enhanced image
    print_step(3, "Saving enhanced image...")
    base, ext = os.path.splitext(config.IMAGE_PATH)
    enhanced_image_path = f"{base}_enhanced{ext}"
    enhanced_image.save(enhanced_image_path)

    # Step 4: Translate the caption
    print_step(4, "Translating captions...")
    multilingual_captions = translate_caption(
        english_caption,
        target_languages=config.TARGET_LANGUAGES
    )

    # Step 5: Finalize output
    print_step(5, f"Finalizing output (Mode: {output_mode})...")
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