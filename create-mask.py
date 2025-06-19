"""
Script de python para la creación de daños usando ruido Perlin.
Uso: python create-mask.py --input_root ruta/a/dataset --output_root ruta/a/carpeta_salida
"""

import cv2
import numpy as np
import os
import random
from noise import pnoise2


###########################################
# Función para guardar cada imagen dañada #
###########################################
def damage_image(image_path, output_basepath):

    image = cv2.imread(image_path)
    
    # Aplicar pipeline
    final_damaged = apply_damage_pipeline(image)

    # Guardar imagen
    cv2.imwrite(output_basepath + ".jpg", final_damaged)


#####################################################
# Función para iterar sobre cada imagen del dataset #
#####################################################

def batch_damage_images(input_root, output_root):

    # El directorio input_root contiene una carpeta por artista del cual se quiere dañar sus obras
    for subfolder in os.listdir(input_root):
        subfolder_path = os.path.join(input_root, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # Guardamos con el mismo nombre la carpeta del artista en output_root
        output_subfolder = os.path.join(output_root, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        # Iteramos sobre cada imagen dentro de la carpeta de artista
        for file_name in os.listdir(subfolder_path):
            if file_name.lower().endswith(".jpg"):
                input_image_path = os.path.join(subfolder_path, file_name)
                base_name = file_name.replace(".jpg", "")
                
                # Preparamos directorio de guardado de la imagen dañada como output_subfolder/damaged_imagen.jpg
                output_basepath = os.path.join(output_subfolder, f"damaged_{base_name}")
                damage_image(input_image_path, output_basepath)
             


#####################################
# Función para aplicar ruido Perlin #
#####################################
   
def generate_damage_mask(image, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, threshold=0.65):
    
    h, w = image.shape[:2] # Sacamos numero de pixeles en imagen en height x width
    noise_img = np.zeros((h, w), dtype=np.float32) # Inicializamos una matriz de ceros
    
    # Desplazamiento random para que el daño del ruido Perlin se mueva de obra a obra
    x_offset = random.uniform(0, 200.0)
    y_offset = random.uniform(0, 200.0)
    
    # Iteramos sobre la matriz de píxeles de la imagen
    for y in range(h):
        for x in range(w):
            # pnoise2 = perlin noise
            noise_img[y, x] = pnoise2((x + x_offset) / scale,
                                      (y + y_offset) / scale,
                                      octaves=octaves,
                                      persistence=persistence,
                                      lacunarity=lacunarity)
    # Normalizamos a 0..1
    minv, maxv = noise_img.min(), noise_img.max()
    noise_norm = (noise_img - minv) / (maxv - minv)
    
    # Creamos la máscara de daño. Definimos el límite, a partir del cual los picos del ruido Perlin serán daños
    mask = (noise_norm > threshold).astype(np.uint8)
    
    # Creamos imagen con color canvas
    canvas_color = np.array([255, 255, 255], dtype=np.uint8)
    # Expandimos la máscara a 3 canales para BGR
    mask_3c = mask[:, :, None]
    # Creamos capa uniforme del color canvas
    canvas_layer = np.zeros_like(image)
    canvas_layer[:] = canvas_color
    
    # Aplicamos máscara: píxeles originales donde mask==0, máscara color canvas donde mask==1
    final_img = image * (1 - mask_3c) + canvas_layer * mask_3c
    
    return final_img


################################
# Función para generar grietas #
################################

def generate_crack_patterns(image, num_cracks=3, max_length=250, thickness_range=(1, 2)):
    damaged = image.copy()
    h, w = damaged.shape[:2]
    
    # Bucle para generar la cantidad de grietas especificada
    for _ in range(num_cracks):
        
        # Elegimos un punto de inicio aleatorio dentro de la imagen
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        # Elegimos un número aleatorio de grietas
        num_segments = np.random.randint(15, 30)
        # Elegimos un ángulo inicial aleatorio para las grietas
        angle = np.random.uniform(0, 2*np.pi)
        points = [(x, y)]
        
        # Cada segmento de la grieta tendrá la misma longitud
        segment_length = max_length / num_segments
        
        # Bucle para generar la grieta por numero de segmentos
        for _ in range(num_segments):
            # Variamos ligeramente el ángulo para que la grieta no sea recta
            angle += np.random.uniform(-0.5, 0.5)
             
            dx = int(segment_length * np.cos(angle) + np.random.randint(-5, 5))
            dy = int(segment_length * np.sin(angle) + np.random.randint(-5, 5))
            
            # Corregimos para no salirnos de la imagen
            x = np.clip(x + dx, 0, w - 1)
            y = np.clip(y + dy, 0, h - 1)
            
            points.append((x, y))
            
        # Color de la grieta
        color = (199, 231, 239)
        thickness = random.randint(thickness_range[0], thickness_range[1])
        cv2.polylines(damaged, [np.array(points, dtype=np.int32)], isClosed=False, color=color, thickness=thickness)
    return damaged

#################################
# Función para aplicar pipeline #
#################################

def apply_damage_pipeline(image):
    # Primer paso: Generamos imagen dañada con perlin noise
    perlin_damaged = generate_damage_mask(image)
    
    # Segundo paso: Generar grietas sobre imagen dañada
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