# utils.py - Utility functions
import re
import platform
import subprocess

def clean_markdown_for_tts(text):
    """Remove or convert markdown formatting for better TTS output"""
    # Replace **bold** with actual text without asterisks
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    
    # Replace *italic* with actual text without asterisks
    text = re.sub(r'\*([^*]+?)\*', r'\1', text)
    
    # Remove markdown links and just keep the text
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    
    # Handle other markdown elements as needed
    # Remove hashtags for headers
    text = re.sub(r'#+\s+', '', text)
    
    # Remove bullet points
    text = re.sub(r'^-\s+', '', text, flags=re.MULTILINE)
    
    return text

def generate_full_text_query(input_text: str) -> str:
    """Generate a full text query from input text"""
    cleaned = ' '.join([word for word in input_text.split() if word])
    return ' AND '.join([f"{word}~2" for word in cleaned.split()])

def play_audio(output_filepath):
    """Play audio file based on the operating system"""
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', output_filepath])
        elif os_name == "Windows":  # Windows
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
        elif os_name == "Linux":  # Linux
            subprocess.run(['aplay', output_filepath])  # Alternative: use 'mpg123' or 'ffplay'
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")
        return False
    return True