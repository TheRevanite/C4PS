import os
import requests
import argostranslate.package
import argostranslate.translate
from . import config

def download_assets():
    """
    Downloads all necessary assets: model weights and language packages.
    """
    os.makedirs(config.ASSETS_DIR, exist_ok=True)
    os.makedirs(config.WEIGHTS_DIR, exist_ok=True)

    if not os.path.exists(config.IMAGE_PATH):
        try:
            response = requests.get(config.IMAGE_URL)
            response.raise_for_status()
            with open(config.IMAGE_PATH, "wb") as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Error downloading image: {e}")

    if not os.path.exists(config.ENHANCER_WEIGHTS_PATH):
        try:
            response = requests.get(config.ENHANCER_WEIGHTS_URL, stream=True)
            response.raise_for_status()
            with open(config.ENHANCER_WEIGHTS_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Error downloading weights: {e}")

    try:
        if argostranslate.translate.get_translation_from_codes("en", "fr"):
            return
    except Exception:
        pass

    try:
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()

        source_lang_code = "en"
        target_lang_codes = config.TARGET_LANGUAGES #['fr', 'es', 'de']

        for lang_code in target_lang_codes:
            package_to_install = next(
                filter(
                    lambda x: x.from_code == source_lang_code and x.to_code == lang_code, available_packages
                )
            )
            argostranslate.package.install_from_path(package_to_install.download())
    except Exception as e:
        print(f"[ERROR] Error installing Argos Translate models: {e}")
        print("[INFO] Please check your internet connection.")
