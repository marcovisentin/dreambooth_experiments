import os
import numpy as np
from PIL import Image, ImageOps
import random

def get_bounding_box(mask, max_expand=50):
    """
    This function gets the bounding box around the non-zero regions of the mask with some variation in size.
    """
    mask_array = np.array(mask)
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    xmin_expand = max(0, xmin - random.randint(0, max_expand))
    ymin_expand = max(0, ymin - random.randint(0, max_expand))
    xmax_expand = min(xmax + random.randint(0, max_expand), mask.shape[1])
    ymax_expand = min(ymax + random.randint(0, max_expand), mask.shape[0])

    return xmin_expand, ymin_expand, xmax_expand, ymax_expand

def create_mask(image_path, output_dir):
    """
    This function create a mask from an image with white backgroud.
    """
    image_name = image_path.split('/')[-1]
    mask_path = os.path.join(output_dir, image_name.replace("nobkg", "mask"))
    
    image = Image.open(image_path).convert("RGBA")
    image_array = np.array(image)
    mask_array = (image_array[:,:,3] > 0).astype(np.uint8)
    xmin_expand, ymin_expand, xmax_expand, ymax_expand = get_bounding_box(mask_array)

    # Bounding box mask
    expand_mask_array = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)
    expand_mask_array[ymin_expand:ymax_expand+1, xmin_expand:xmax_expand+1] = 255
    new_mask = Image.fromarray(expand_mask_array, mode='L')
    new_mask.save(mask_path)
    print("Created and saved {}".format(mask_path))


if __name__=="__main__":

    images_dir = "/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/tuning_images/angry_bird_speaker_inpaint"
    output_dir = "/home/marco/Desktop/ARvertise/code/diffusers/examples/dreambooth/images/tuning_images/angry_bird_speaker_inpaint"
    images = [image for image in os.listdir(images_dir) if "nobkg" in image]
    
    for image in images:
        
        image_path = os.path.join(images_dir, image)
        create_mask(image_path, output_dir)
    
    

