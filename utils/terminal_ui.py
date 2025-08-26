import os
from PIL import Image
import warnings

#ANSI escape codes for colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    """Clears the terminal screen for a clean UI."""
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS and Linux
    else:
        _ = os.system('clear')

def print_header():
    """Prints the main application header."""
    print(f"{Colors.BOLD}{Colors.CYAN}C4PS: Captioning + Augmentation + Processing for Socials{Colors.ENDC}")
    print("-" * 60)

def print_step(step_num, message):
    """Prints a formatted step message."""
    print(f"{Colors.BLUE}Step {step_num}:{Colors.ENDC} {message}")

def display_offline_report(enhanced_path, english_caption, multilingual_captions):
    """Displays the final report in the terminal and opens the image."""
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}{Colors.HEADER}--- C4PS Offline Report ---{Colors.ENDC}")
    print("=" * 60)
    
    print(f"\n{Colors.CYAN}Enhanced Image:{Colors.ENDC}")
    print(f"  Saved to: {enhanced_path}")
    
    print(f"\n{Colors.CYAN}English Caption:{Colors.ENDC}")
    print(f"  {Colors.GREEN}{english_caption}{Colors.ENDC}")
    
    print(f"\n{Colors.CYAN}Multilingual Captions:{Colors.ENDC}")
    for lang, text in multilingual_captions.items():
        print(f"  - {Colors.BOLD}{lang.upper()}:{Colors.ENDC} {text}")
        
    print("\n" + "=" * 60)
    
    try:
        print("Opening enhanced image...")
        image = Image.open(enhanced_path)
        image.show()
    except Exception as e:
        print(f"{Colors.FAIL}Could not open image automatically: {e}{Colors.ENDC}")

def display_online_report(message_url, server_invite):
    """Displays the final online report in the terminal."""
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}{Colors.HEADER}--- C4PS Online Report ---{Colors.ENDC}")
    print("=" * 60)
    
    print(f"\n{Colors.GREEN}Results sent to Discord successfully!{Colors.ENDC}")
    
    print(f"\n{Colors.CYAN}View the message here:{Colors.ENDC}")
    print(f"  {Colors.UNDERLINE}{message_url}{Colors.ENDC}")
    
    if server_invite:
        print(f"\n{Colors.CYAN}Not in the server yet? Join here:{Colors.ENDC}")
        print(f"  {Colors.UNDERLINE}{server_invite}{Colors.ENDC}")
        
    print("\n" + "=" * 60)

def suppress_warnings():
    """Suppresses common warnings from imported libraries for a cleaner UI."""
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')
