import numpy as np
from PIL import Image
from skimage import io, measure, draw, filters
from scipy.ndimage import label, distance_transform_edt
import random
import os


# Input folder containing images
content_folder = './Input_images'

# Getting the list of images in the folder
files = os.listdir(content_folder)
image_paths = [os.path.join(content_folder, image_file) for image_file in files]

# Create output directory if it doesn't exist
output_directory = './Output_images'
os.makedirs(output_directory, exist_ok=True)


# Function to calculate the sign of the determinant of three points (used for orientation check)
def sgn(p1, p2, p3):
    matrix = np.array([
        [p1[0], p1[1], 1],
        [p2[0], p2[1], 1],
        [p3[0], p3[1], 1]
    ])
    return np.linalg.det(matrix)

# Function to check if a point is convex (white vertex)
def is_convex(p1, p2, p3):
    return sgn(p1, p2, p3) > 0

# Function to check if a point is concave (black vertex)
def is_concave(p1, p2, p3):
    return sgn(p1, p2, p3) < 0


def process_image(image_path, output_directory):
    def chamfer_labeling(binary_image):
        labels = np.zeros(binary_image.shape, dtype=int)
        current_label = 1

        def neighbors(x, y):
            return [(x-1, y), (x+1, y), (x, y-1), (x, y+1),
                    (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)]

        def bfs(start_x, start_y):
            queue = [(start_x, start_y)]
            labels[start_x, start_y] = current_label
            while queue:
                x, y = queue.pop(0)
                for nx, ny in neighbors(x, y):
                    if (0 <= nx < binary_image.shape[0] and 0 <= ny < binary_image.shape[1] and
                        binary_image[nx, ny] == 255 and labels[nx, ny] == 0):
                        labels[nx, ny] = current_label
                        queue.append((nx, ny))
                        labels[nx, ny] = current_label

        for x in range(binary_image.shape[0]):
            for y in range(binary_image.shape[1]):
                if binary_image[x, y] == 255 and labels[x, y] == 0:
                    bfs(x, y)
                    current_label += 1

        return labels, current_label - 1

    def compute_mpp(binary_image):
        labeled_image, num_features = chamfer_labeling(binary_image)
        mpp_polygons = []

        for label_id in range(1, num_features + 1):
            component_mask = labeled_image == label_id
            contours = measure.find_contours(component_mask, 0.5, fully_connected="low")
            
            for contour in contours:
                perimeter = np.sum(np.sqrt(np.diff(contour[:, 0])**2 + np.diff(contour[:, 1])**2))
                mpp_polygon = measure.approximate_polygon(contour, 0.01 * perimeter)
                mpp_polygons.append(mpp_polygon)

        return labeled_image, mpp_polygons

    image = io.imread(image_path, as_gray=True)
    binary_image = (image > filters.threshold_otsu(image)).astype(np.uint8) * 255
    labels, mpp_polygons = compute_mpp(binary_image)

    colored_labels = np.zeros((*binary_image.shape, 3), dtype=np.uint8)
    for feature_id in range(1, np.max(labels) + 1):
        mask = labels == feature_id
        random_color = [random.randint(125, 255) for _ in range(3)]
        colored_labels[mask] = random_color

    polyline_image = colored_labels.copy()
    for mpp_polygon in mpp_polygons:
        rr, cc = draw.polygon_perimeter(mpp_polygon[:, 0], mpp_polygon[:, 1], shape=binary_image.shape)
        polyline_image[rr, cc] = [255, 0, 0] 

        for point in mpp_polygon:
            rr, cc = draw.disk((point[0], point[1]), radius=2.5, shape=binary_image.shape)
            polyline_image[rr, cc] = [0, 200, 0]  

    # Save the final image with MPP
    filename = os.path.basename(image_path)
    filename_without_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(output_directory, f"{filename_without_ext}_MPP.bmp")
    Image.fromarray(polyline_image).save(output_path)


for image_path in image_paths:
    process_image(image_path, output_directory)