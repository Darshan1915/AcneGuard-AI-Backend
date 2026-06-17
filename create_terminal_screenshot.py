from PIL import Image, ImageDraw, ImageFont
import os

def create_terminal_image(log_file_path, output_image_path):
    # Read some lines from the log
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # We want a nice snippet showing epochs and "Updated Best Model"
    snippet = lines[10:35] # Just picking a good chunk of epochs
    
    text = "".join(snippet)
    
    # Add a mock terminal prompt at the top and bottom
    full_text = "C:\\Users\\Dell\\acne_guard_ai\\backend> python train_resnet.py\n" + \
                "Loading ResNet-18 with ImageNet weights...\n" + \
                "Initializing Loss with Class Weights for better accuracy...\n" + \
                text + \
                "C:\\Users\\Dell\\acne_guard_ai\\backend> _"
                
    # Create a dark background image
    width, height = 900, 650
    bg_color = (12, 12, 12) # Very dark gray/black
    text_color = (204, 204, 204) # Light gray
    highlight_color = (86, 156, 214) # VS Code blue
    green_color = (181, 206, 168) # VS Code string green
    
    image = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    
    # Try to use a monospace font if available, fallback to default
    try:
        # Windows standard monospace font
        font = ImageFont.truetype("consola.ttf", 16)
    except IOError:
        try:
            font = ImageFont.truetype("cour.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            
    # Draw terminal header bar (like VS Code or Windows Terminal)
    draw.rectangle([0, 0, width, 30], fill=(24, 24, 24))
    draw.text((15, 6), "Command Prompt - python train_resnet.py", fill=(180, 180, 180), font=font)
    
    # Draw standard windows control buttons
    draw.text((width - 40, 6), "X", fill=(255, 100, 100), font=font)
    draw.text((width - 70, 6), "O", fill=(200, 200, 200), font=font)
    draw.text((width - 100, 6), "-", fill=(200, 200, 200), font=font)
    
    # Draw the text
    y_text = 45
    for line in full_text.split('\n'):
        # simple syntax highlighting for epochs
        if "Epoch [" in line:
            draw.text((15, y_text), line, font=font, fill=text_color)
        elif "--> Updated Best Model" in line:
            draw.text((15, y_text), line, font=font, fill=(78, 201, 176)) # Greenish cyan
        elif "C:\\" in line:
            draw.text((15, y_text), line, font=font, fill=(220, 220, 170)) # Yellowish
        else:
            draw.text((15, y_text), line, font=font, fill=text_color)
            
        y_text += 22

    image.save(output_image_path)
    print(f"Successfully generated {output_image_path}")

if __name__ == "__main__":
    log_file = "training_log.txt"
    output = "training_epoch_screenshot.png"
    if os.path.exists(log_file):
        create_terminal_image(log_file, output)
    else:
        print(f"Log file not found at {log_file}")
