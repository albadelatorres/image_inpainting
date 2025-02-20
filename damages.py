import cv2
import numpy as np
import os
import random

def add_random_scratches(image, num_scratches=3, max_length=200, max_deviation=5, thickness_range=(1, 3)):
    """
    Adds long random scratches with irregular, curvy segments.
    The scratch path is constructed as a series of small, randomly deviated segments.
    Increased max_length and number of segments yield a much longer scratch.
    """
    damaged = image.copy()
    h, w = damaged.shape[:2]
    for _ in range(num_scratches):
        # Start at a random point
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        # Increase the number of segments for a longer, more detailed scratch
        num_points = np.random.randint(10, 20)
        angle = np.random.uniform(0, 2*np.pi)
        points = [(x, y)]
        segment_length = max_length / num_points
        for _ in range(num_points):
            # Slight random change in the angle for a natural curve
            angle += np.random.uniform(-0.3, 0.3)
            dx = int(segment_length * np.cos(angle) + np.random.randint(-max_deviation, max_deviation))
            dy = int(segment_length * np.sin(angle) + np.random.randint(-max_deviation, max_deviation))
            x = np.clip(x + dx, 0, w - 1)
            y = np.clip(y + dy, 0, h - 1)
            points.append((x, y))
        # Use a light color for the scratch
        color = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))
        thickness = random.randint(thickness_range[0], thickness_range[1])
        cv2.polylines(damaged, [np.array(points, dtype=np.int32)], isClosed=False, color=color, thickness=thickness)
    # Apply a slight blur to soften the scratches
    damaged = cv2.GaussianBlur(damaged, (3, 3), 0)
    return damaged

def add_random_spots(image, num_spots=8, max_radius=100):
    """
    Adds spots with soft edges simulating stains or localized fading.
    Spots may be circular or elliptical with a Gaussian–blurred mask for a natural gradient.
    """
    damaged = image.copy()
    h, w = damaged.shape[:2]
    overlay = damaged.copy()
    
    for _ in range(num_spots):
        center = (np.random.randint(0, w), np.random.randint(0, h))
        # Increase the minimum radius for larger spots
        radius = np.random.randint(10, max_radius)
        if random.random() < 0.5:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
        else:
            axis1 = np.random.randint(radius // 2, radius)
            axis2 = np.random.randint(radius // 2, radius)
            angle = np.random.randint(0, 360)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, center, (axis1, axis2), angle, 0, 360, 255, -1)
        
        # Create soft edges with Gaussian blur
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Apply a uniform spot color
        spot_color = np.array([random.randint(50, 200) for _ in range(3)], dtype=np.uint8)
        for c in range(3):
            overlay[:, :, c] = np.where(mask == 255, spot_color[c], overlay[:, :, c])
        
        # Create a noise image
        noise = np.random.randint(0, 30, (h, w, 3), dtype='uint8')
        # Compute a weighted blend for the entire image
        blended = cv2.addWeighted(overlay, 0.7, noise, 0.3, 0)
        # Use np.where to replace only the spot region (expand mask to 3 channels)
        overlay = np.where(mask[..., np.newaxis] > 128, blended, overlay)
    
    # Blend the overlay with the original image
    damaged = cv2.addWeighted(damaged, 0.8, overlay, 0.2, 0)
    return damaged

def add_cracks(image, num_cracks=4, max_length=250, thickness_range=(1, 2)):
    """
    Simulates realistic cracks using fractal-like polylines.
    The crack is generated as a series of small segments with larger random deviations.
    """
    damaged = image.copy()
    h, w = damaged.shape[:2]
    
    for _ in range(num_cracks):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        num_segments = np.random.randint(10, 20)
        angle = np.random.uniform(0, 2*np.pi)
        points = [(x, y)]
        segment_length = max_length / num_segments
        for _ in range(num_segments):
            angle += np.random.uniform(-0.5, 0.5)
            dx = int(segment_length * np.cos(angle) + np.random.randint(-3, 3))
            dy = int(segment_length * np.sin(angle) + np.random.randint(-3, 3))
            x = np.clip(x + dx, 0, w - 1)
            y = np.clip(y + dy, 0, h - 1)
            points.append((x, y))
        # Use a dark color to mimic deep cracks
        color = (random.randint(0, 60), random.randint(0, 60), random.randint(0, 60))
        thickness = random.randint(thickness_range[0], thickness_range[1])
        cv2.polylines(damaged, [np.array(points, dtype=np.int32)], isClosed=False, color=color, thickness=thickness)
    return damaged

def remove_random_patch(image, scale_min, scale_max):
    """
    Removes an irregular patch sized as a fraction of the image dimensions.
    scale_min and scale_max determine the minimum and maximum patch radius
    as a fraction of the smaller image dimension.
    """
    damaged = image.copy()
    h, w = damaged.shape[:2]
    
    # Calculate patch radii based on image size (using the smaller of width or height)
    min_dim = min(w, h)
    min_radius = int(min_dim * scale_min)
    max_radius = int(min_dim * scale_max)
    
    # Ensure that there is room for the patch center:
    if w - 2 * max_radius > 0:
        center_x = np.random.randint(max_radius, w - max_radius)
    else:
        center_x = w // 2
    if h - 2 * max_radius > 0:
        center_y = np.random.randint(max_radius, h - max_radius)
    else:
        center_y = h // 2
    
    # Use more vertices for a more irregular shape
    num_vertices = np.random.randint(12, 22)
    angles = np.sort(np.random.uniform(0, 2*np.pi, num_vertices))
    radii = np.random.randint(min_radius, max_radius, num_vertices)
    
    points = []
    for angle, radius in zip(angles, radii):
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        points.append([x, y])
    points = np.array(points, np.int32).reshape((-1, 1, 2))
    
    canvas_color = (220, 220, 220)
    cv2.fillPoly(damaged, [points], canvas_color)
    cv2.polylines(damaged, [points], isClosed=True, color=(180, 180, 180), thickness=1)
    
    return damaged

def damage_image(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image:", image_path)
        return
    
    damaged = add_random_scratches(image, num_scratches=5, max_length=200, max_deviation=8, thickness_range=(1, 3))
    damaged = add_random_spots(damaged, num_spots=8, max_radius=40)
    damaged = add_cracks(damaged, num_cracks=3, max_length=150, thickness_range=(1, 2))
    damaged = remove_random_patch(damaged, scale_min=0.1, scale_max=0.25)
    
    cv2.imwrite(output_path, damaged)

if __name__ == '__main__':
    # Directory with original images
    input_dir = '/Users/albadelatorres/Desktop/TFG/archive/musemart/dataset_updated/training_set/iconography_shortened'
    # Output directory for damaged images
    output_dir = '/Users/albadelatorres/Desktop/TFG/archive/musemart/dataset_updated/training_set/iconography_shortened_damaged'
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(('jpg', 'jpeg', 'png')):
            image_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, f"damaged_{file_name}")
            damage_image(image_path, output_path)