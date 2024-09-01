import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Input folder containing images
content_folder = './Input_images'

# Getting the list of images in the folder
files = os.listdir(content_folder)
image_paths = [os.path.join(content_folder, image_file) for image_file in files]

# Create output directory if it doesn't exist
output_directory = './Output_images'
os.makedirs(output_directory, exist_ok=True)

for image_path in image_paths:
    
    # Load the image and convert to grayscale numpy array
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    
    # Contour detection
    contours = []
    visited = np.zeros_like(image_array, dtype=bool)

    def is_border(x, y):
        if image_array[x, y] == 0:
            return False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if 0 <= x + dx < image_array.shape[0] and 0 <= y + dy < image_array.shape[1]:
                    if image_array[x + dx, y + dy] == 0:
                        return True
        return False

    def follow_contour(start_x, start_y):
        contour = []
        x, y = start_x, start_y
        start_dir = 0
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        while True:
            contour.append((x, y))
            visited[x, y] = True
            found_next = False
            for i in range(8):
                dx, dy = directions[(start_dir + i) % 8]
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < image_array.shape[0] and 0 <= new_y < image_array.shape[1]:
                    if is_border(new_x, new_y) and not visited[new_x, new_y]:
                        x, y = new_x, new_y
                        start_dir = (start_dir + i + 6) % 8  # Set new direction
                        found_next = True
                        break
            if not found_next:
                break
        return contour

    for x in range(image_array.shape[0]):
        for y in range(image_array.shape[1]):
            if is_border(x, y) and not visited[x, y]:
                contour = follow_contour(x, y)
                contours.append(contour)

    # Draw contours
    contour_image = np.zeros_like(image_array)
    for contour in contours:
        for (x, y) in contour:
            contour_image[x, y] = 255  # Mark the contour with white color
    
    filename = os.path.basename(image_path)
    filename_without_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(output_directory, f"{filename_without_ext}_contours.bmp")
    Image.fromarray(contour_image).save(output_path)