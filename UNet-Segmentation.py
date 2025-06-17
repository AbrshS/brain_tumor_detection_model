import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_data(data_path, mask_path, img_size=(256, 256)):
    images = []
    masks = []
    
    # Load images and masks with reduced precision
    for img_name in os.listdir(data_path):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(data_path, img_name)
            mask_name = img_name.replace('.jpg', '_mask.png')
            mask_img_path = os.path.join(mask_path, mask_name)
            
            if os.path.exists(mask_img_path):
                # Load and process image with reduced precision
                img = Image.open(img_path).convert('L')
                img = img.resize(img_size)
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Load and process mask with reduced precision
                mask = Image.open(mask_img_path).convert('L')
                mask = mask.resize(img_size)
                mask_array = np.array(mask, dtype=np.float32) / 255.0
                
                images.append(img_array)
                masks.append(mask_array)
                
                # Free up memory
                del img, mask
    
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)