import cv2
import numpy as np
import os
import random
# Import Worley noise generator for chipping effects
from noise import pnoise2
# Import Worley noise generator for chipping effects
from pythonworley import worley

def damage_image(image_path, output_basepath):
    """
    Generates and saves a final damaged image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error al cargar la imagen:", image_path)
        return

    # Apply the damage pipeline to simulate realistic damage
    final_damaged = apply_damage_pipeline(image)

    # Save only the final damaged image
    cv2.imwrite(output_basepath + ".jpg", final_damaged)


def batch_damage_images(input_root, output_root):
    """
    Processes images in the input directory and saves the final damaged image in the output directory.
    """
    for subfolder in os.listdir(input_root):
        subfolder_path = os.path.join(input_root, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # Use the same folder name for damaged images
        output_subfolder = os.path.join(output_root, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        for file_name in os.listdir(subfolder_path):
            if file_name.lower().endswith(".jpg"):
                input_image_path = os.path.join(subfolder_path, file_name)
                base_name = file_name.replace(".jpg", "")
                output_basepath = os.path.join(output_subfolder, f"damaged_{base_name}")
                damage_image(input_image_path, output_basepath)
                
def generate_damage_mask(image, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, threshold=0.65):
    """Generate a binary mask for damage regions using multi-octave Perlin noise."""
    h, w = image.shape[:2] # sacamos numero de pixeles en imagen en height x width
    noise_img = np.zeros((h, w), dtype=np.float32) # inicializamos una matriz de ceros
    # Random offsets so each image gets a unique Perlin pattern
    x_offset = random.uniform(0, 200.0)
    y_offset = random.uniform(0, 200.0)
    
    #iteramos sobre la matriz de píxeles de la imagen
    for y in range(h):
        for x in range(w):
            # pnoise2= perlin noise
            noise_img[y, x] = pnoise2((x + x_offset) / scale,
                                      (y + y_offset) / scale,
                                      octaves=octaves,
                                      persistence=persistence,
                                      lacunarity=lacunarity)
    # normalize to 0..1
    minv, maxv = noise_img.min(), noise_img.max()
    noise_norm = (noise_img - minv) / (maxv - minv)
    # threshold to binary mask
    mask = (noise_norm > threshold).astype(np.uint8)
    
    # crear imagen con color canvas
    canvas_color = np.array([199, 231, 239], dtype=np.uint8)
    # Expande la máscara a 3 canales para BGR
    mask_3c = mask[:, :, None]
    # Crea capa uniforme del color canvas
    canvas_layer = np.zeros_like(image)
    canvas_layer[:] = canvas_color
    # Combina: píxeles originales donde mask==0, canvas donde mask==1
    final_img = image * (1 - mask_3c) + canvas_layer * mask_3c
    return final_img

# Worley noise generator for chipping effects
def generate_worley_noise(image):
    """Generate a Worley noise texture for chipping effects."""
    h, w = image.shape[:2]
    dens=128
    shape=(8,4)
    # Generate Worley noise distances and centers
    w_noise, c = worley(shape, dens=dens, seed=0)
    # Use first channel (closest feature distance) and transpose to match image axes
    #bubble pattern: 
    w_map = w_noise[0].T
    # cobblestone pattern: 
    #w_map = w_noise[1].T - w_noise[0].T
    # Resize to fit the image dimensions
    w_resized = cv2.resize(w_map, (w, h), interpolation=cv2.INTER_LINEAR)
    # Normalize to 0..1
    minv, maxv = w_resized.min(), w_resized.max()
    w_norm = (w_resized - minv) / (maxv - minv)
    # Create 3-channel grayscale BGR noise
    worley_bgr = (np.stack([w_norm]*3, axis=2) * 255).astype(np.uint8)
        # crear imagen con color canvas
    canvas_color = np.array([199, 231, 239], dtype=np.uint8)
    # Normalize worley noise to 0..1 for blending
    worley_norm = worley_bgr.astype(np.float32) / 255.0
    # Create a uniform canvas layer
    canvas_layer = np.zeros_like(image)
    canvas_layer[:] = canvas_color
    # Blend the input image and the canvas layer using the worley mask
    # Convert image to float for blending
    image_f = image.astype(np.float32)
    final_image = ((image_f * (1 - worley_norm)) + canvas_layer.astype(np.float32) * worley_norm).astype(np.uint8)
    return final_image


def generate_crack_patterns(image, num_cracks=3, max_length=250, thickness_range=(1, 2)):
    """Overlay crack-like patterns on the given image via a procedural algorithm."""
    damaged = image.copy()
    h, w = damaged.shape[:2]
    for _ in range(num_cracks):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        num_segments = np.random.randint(15, 30)
        angle = np.random.uniform(0, 2*np.pi)
        points = [(x, y)]
        segment_length = max_length / num_segments
        for _ in range(num_segments):
            angle += np.random.uniform(-0.5, 0.5)
            dx = int(segment_length * np.cos(angle) + np.random.randint(-5, 5))
            dy = int(segment_length * np.sin(angle) + np.random.randint(-5, 5))
            x = np.clip(x + dx, 0, w - 1)
            y = np.clip(y + dy, 0, h - 1)
            points.append((x, y))
        # off-white canvas crack color
        color = (199, 231, 239)
        thickness = random.randint(thickness_range[0], thickness_range[1])
        cv2.polylines(damaged, [np.array(points, dtype=np.int32)], isClosed=False, color=color, thickness=thickness)
    return damaged


def apply_damage_pipeline(image):
    """Apply the complete damage simulation pipeline on the image."""
    # Primer paso: Generamos imagen dañada con perlin noise
    perlin_damaged = generate_damage_mask(image)
    
    # Segundo paso: Generamos imagen dañada con worley noise
    #worley_damaged = generate_worley_noise(perlin_damaged)
    
    # Overlay procedural crack patterns
    with_cracks = generate_crack_patterns(perlin_damaged)
    
    return with_cracks

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Genera imagen dañada.")
    parser.add_argument("--input_root", type=str, required=True,
                        help="Carpeta con subcarpetas (por artista). Ej: training/resized")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Carpeta de salida. Ej: training/damaged-resized-masked")
    args = parser.parse_args()
    
    os.makedirs(args.output_root, exist_ok=True)
    batch_damage_images(args.input_root, args.output_root)
    print(f"Generado en {args.output_root}")