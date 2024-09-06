import numpy as np
from skimage import io, measure, draw, filters
from scipy.ndimage import label
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

# Input folder containing images
content_folder = './Input_images'

# Getting the list of images in the folder
files = os.listdir(content_folder)
image_paths = [os.path.join(content_folder, image_file) for image_file in files]

def process_and_display_images(image_paths):
    
    # Create output directory if it doesn't exist
    output_directory = './Output_images'
    os.makedirs(output_directory, exist_ok=True)
    
    for image_path in image_paths:
        
        image = io.imread(image_path, as_gray=True)
        binary_image = (image > filters.threshold_otsu(image)).astype(np.uint8) * 255
        

        
        # Chamfer Distance Transform
        height, width = binary_image.shape
        distance_map = np.full((height, width), np.inf)
        
        for row in range(height):
            for col in range(width):
                if binary_image[row, col] == 255:
                    distance_map[row, col] = 0
                else:
                    min_distance = np.inf
                    if row > 0:
                        min_distance = min(min_distance, distance_map[row-1, col] + 1)
                    if col > 0:
                        min_distance = min(min_distance, distance_map[row, col-1] + 1)
                    if row > 0 and col > 0:
                        min_distance = min(min_distance, distance_map[row-1, col-1] + np.sqrt(2))
                    if row > 0 and col < width - 1:
                        min_distance = min(min_distance, distance_map[row-1, col+1] + np.sqrt(2))
                    distance_map[row, col] = min_distance
        
        for row in range(height-1, -1, -1):
            for col in range(width-1, -1, -1):
                if binary_image[row, col] == 255:
                    distance_map[row, col] = 0
                else:
                    min_distance = distance_map[row, col]
                    if row < height - 1:
                        min_distance = min(min_distance, distance_map[row+1, col] + 1)
                    if col < width - 1:
                        min_distance = min(min_distance, distance_map[row, col+1] + 1)
                    if row < height - 1 and col < width - 1:
                        min_distance = min(min_distance, distance_map[row+1, col+1] + np.sqrt(2))
                    if row < height - 1 and col > 0:
                        min_distance = min(min_distance, distance_map[row+1, col-1] + np.sqrt(2))
                    distance_map[row, col] = min_distance
        
        # Label connected components
        labeled_image, num_features = label(binary_image)
        
        # Generate a random color for each label
        labeled_color_image = np.zeros((*binary_image.shape, 3), dtype=np.uint8)
        for label_idx in range(1, num_features + 1):
            mask = labeled_image == label_idx
            color = [random.randint(125, 255) for _ in range(3)]
            labeled_color_image[mask] = color
        
        # Generate output file path
        filename = os.path.basename(image_path)
        filename_without_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_directory, f"{filename_without_ext}_labeled.bmp")
        
        # Save the labeled color image
        labeled_color_image_pil = Image.fromarray(labeled_color_image)
        labeled_color_image_pil.save(output_path)
        
        
process_and_display_images(image_paths)
