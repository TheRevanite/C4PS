import os
from datetime import datetime
from discord import SyncWebhook, File, Embed
from .terminal_ui import display_online_report, display_offline_report

def send_to_discord(webhook_url, server_invite, enhanced_path, english_caption, multilingual_captions):
    """Sends the results to Discord and displays a report in the terminal."""
    if not webhook_url:
        print("[ERROR] Discord Webhook URL not found in .env file. Skipping online post.")
        return

    try:
        webhook = SyncWebhook.from_url(webhook_url)
        embed = Embed(
            title="Image Captioning Result",
            description=f"**English Caption:**\n> {english_caption}",
            color=5814783
        )
        translations_text = ""
        for lang, text in multilingual_captions.items():
            translations_text += f"**{lang.upper()}:** {text}\n"
        embed.add_field(name="Multilingual Captions", value=translations_text)
        embed.set_footer(text=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        image_file = File(enhanced_path, filename=os.path.basename(enhanced_path))
        
        message = webhook.send(
            file=image_file,
            username="C4PS Pipeline",
            embed=embed,
            wait=True
        )
        
        display_online_report(message.jump_url, server_invite)

    except Exception as e:
        print(f"[ERROR] Failed to send results to Discord: {e}")

