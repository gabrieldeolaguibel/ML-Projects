import base64
import re

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def extract_code(text):
    pattern = r'```(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None