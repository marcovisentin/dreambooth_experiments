import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image
from lang_sam import LangSAM

def segment_image(image_path, text_prompt, output_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image at {image_path}")
        return

    model = LangSAM()
    image_pil = Image.open(image_path).convert("RGB")
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    
    # Save the binary mask
    output_filename = Path(image_path).stem + '_mask.png'
    mask = masks[0].detach().cpu().numpy()
    cv2.imwrite(str(output_path / output_filename), mask * 255)
    print(f"Saved binary mask to {output_filename}")

def process_folder(folder_path, text_prompt):
    # Create output directory
    output_path = Path(folder_path) / "masks"
    output_path.mkdir(exist_ok=True)

    # Process each image in the directory
    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            segment_image(str(Path(folder_path) / image_file), text_prompt, output_path)

if __name__ == "__main__":

    process_folder("/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/tuning_images/dreambooth_dataset/clock", text_prompt = "clock")
