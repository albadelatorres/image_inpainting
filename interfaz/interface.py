from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import torch
import piq


app = Flask(__name__)

def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)

def calculate_mse(img1, img2):
    return mean_squared_error(img1.flatten(), img2.flatten())

def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2)

def calculate_orb(img1, img2):
    # Convert to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kpA, desA = orb.detectAndCompute(img1, None)
    kpB, desB = orb.detectAndCompute(img2, None)

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desA, desB)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    return len(matches), matches

def calculate_fsim(img1_bgr, img2_bgr):
    """
    FSIM con PIQ (valor en [0,1]) usando imágenes RGB normalizadas.
    Evitamos la aserción manteniendo 3 canales.
    """
    rgb1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
    t1 = torch.from_numpy(rgb1).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # [1,3,H,W]
    t2 = torch.from_numpy(rgb2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        value = piq.fsim(t1, t2, data_range=1.0)  # chromatic=True por defecto
    return value.item()

# --- New helper functions for LPIPS and DISTS ---
def calculate_lpips(img1_bgr, img2_bgr):
    """
    LPIPS distancia perceptual (↓ mejor).  Utiliza implementación funcional de PIQ (red VGG por defecto).
    """
    rgb1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
    t1 = torch.from_numpy(rgb1).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # [1,3,H,W]
    t2 = torch.from_numpy(rgb2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        lpips_metric = piq.LPIPS()
        value = lpips_metric(t1,t2)
        #value = lpips_metric(t1, t2)
    return value.item()


def calculate_dists(img1_bgr, img2_bgr):
    """
    DISTS: Deep Image Structure and Texture Similarity (↓ mejor).
    """
    rgb1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
    t1 = torch.from_numpy(rgb1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t2 = torch.from_numpy(rgb2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        dists = piq.DISTS()
        value = dists(t1,t2)
    return value.item()
    
# Cargamos los modelos
autoencoder_impresionistas = load_model("gan_impresionistas.h5", compile=False)
autoencoder_iconografia = load_model("gan_iconografia.keras", compile=False)
autoencoder_abstracto = load_model("gan_abstracto.keras", compile=False)


@app.route("/", methods=["GET", "POST"])
def index():
    # Lista de imagenes disponibles
    images = os.listdir(os.path.join(app.static_folder, "images"))
    if request.method == "POST":
        # Obtener la imagen subida o del carousel
        uploaded_file = request.files.get("image")
        sample_name   = request.form.get("sample_image")

        img_path=""
        # Opcion para imagen subida
        if uploaded_file and uploaded_file.filename:
            img_path= uploaded_file
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # decodificar bytes del buffer a img

        # Opcion para imagen del carousel
        elif sample_name:
            img_path = os.path.join(app.static_folder, "images", sample_name)
            img_bgr  = cv2.imread(img_path)

        else:
            return render_template("index.html", images=images,
                                   error="ninguna imagen seleccionada")

        # BGR a RGB y preprocesado (equivalente al del modelo)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # BGR -> RGB
        img = cv2.resize(img, (512, 512)) # 512x512
        img = img.astype("float32") / 255.0 # [0,1]
        img = np.expand_dims(img, axis=0) # añadir batch_size (1)

        # Elegimos modelo segun estilo elegido en el dropdown
        estilo = request.form.get("estilo")
        if estilo == "impresionistas":
            print("impresionistas")
            model = autoencoder_impresionistas
        elif estilo == "arte-abstracto":
            print("abstr")
            model = autoencoder_abstracto
        elif estilo == "iconografia":
            print("icon")
            model = autoencoder_iconografia
        elif estilo == "ICT":
            os.chdir("/Users/albadelatorres/Desktop/TFG/interfaz/ICT-main-3")
            ICT_command = "python run.py --input_image " + img_path+ " --input_mask /Users/albadelatorres/Desktop/TFG/impresionistas/training/masks/Hassam_resized --sample_num 1 --save_place /Users/albadelatorres/Desktop/TFG/impresionistas/training/resultados_ICT --ImageNet --visualize_all"
            

        # Predicción
        reconstructed = model.predict(img)
        print(reconstructed.min(), reconstructed.max())
        reconstructed = np.clip(reconstructed[0], 0.0, 1.0) #sacamos prediccion del batch (1,512,512,3) -> (512,512,3)
        reconstructed_img = (reconstructed * 255).astype("uint8") #conversion de [0,1] -> [0,255]

        # Cálculo de MSE y SSIM
        if sample_name:
            original_path= os.path.join(app.static_folder, "original", sample_name)
            original_bgr = cv2.imread(original_path) # sacamos imagen original
            original_uint8 = cv2.resize(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), (512, 512)) #pasamos de BGR a RGB
            mse_val  = (calculate_mse(original_uint8, reconstructed_img) / (255.0 ** 2)) 
            ssim_val = calculate_ssim(original_uint8, reconstructed_img)
            orb_len, _ = calculate_orb(original_uint8, reconstructed_img)
            fsim_val = calculate_fsim(original_uint8, reconstructed_img)
            lpips_val = calculate_lpips(original_uint8, reconstructed_img)
            dists_val = calculate_dists(original_uint8, reconstructed_img)

        # Guardamos el resultado
        output_path = "static/output.png"
        cv2.imwrite(output_path, cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR)) # cv2 espera bgr
        if sample_name:
            return render_template("index.html", images=images, estilo=estilo, mse_val=mse_val, ssim_val=ssim_val, orb_val= orb_len, fsim_val=fsim_val, lpips_val=lpips_val, dists_val=dists_val, output_path=output_path)
        else:
            return render_template("index.html", images=images, estilo=estilo, mse_val=0, ssim_val=0, orb_val= 0, fsim_val=0, lpips_val=0, dists_val=0, output_path=output_path)
    return render_template("index.html", images=images)

if __name__ == "__main__":
    app.run(debug=True)