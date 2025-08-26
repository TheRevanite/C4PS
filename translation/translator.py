import argostranslate.translate
import os

#This suppresses a startup warning from argostranslate
os.environ["ARGOS_DEVICE_TYPE"] = "cuda"

def translate_caption(caption, target_languages=['fr', 'es', 'de']):
    """
    Translates the caption into multiple target languages using the offline Argos Translate.

    Args:
        caption (str): The English caption to translate.
        target_languages (list): A list of language codes.

    Returns:
        dict: A dictionary mapping language codes to translated text.
    """
    translations = {}
    source_lang = "en"

    if not caption or not caption.strip():
        print(f"[WARNING] Skipping translation for empty caption.")
        for lang in target_languages:
            translations[lang] = "Translation skipped."
        return translations

    for target_lang in target_languages:
        try:
            installed_translation = argostranslate.translate.get_translation_from_codes(source_lang, target_lang)
            if installed_translation:
                translated_text = installed_translation.translate(caption)
                translations[target_lang] = translated_text
            else:
                raise RuntimeError(f"Translation model for en -> {target_lang} not found.")
        except Exception as e:
            print(f"[ERROR] Could not translate to {target_lang}: {e}")
            translations[target_lang] = "Translation failed."
            
    return translations
