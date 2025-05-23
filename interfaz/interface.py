from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error


app = Flask(__name__)

def calculate_mse(img1, img2):
    return mean_squared_error(img1.flatten(), img2.flatten())

def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2)

# Cargamos los modelos
autoencoder_impresionistas = load_model("gan_impresionistas.h5", compile=False)
autoencoder_iconografia = load_model("gan_iconografia.h5", compile=False)
autoencoder_abstracto = load_model("gan_abstracto.keras", compile=False)


@app.route("/", methods=["GET", "POST"])
def index():
    # Lista de imagenes disponibles
    images = os.listdir(os.path.join(app.static_folder, "images"))
    if request.method == "POST":
        # Obtener la imagen subida o del carousel
        uploaded_file = request.files.get("image")
        sample_name   = request.form.get("sample_image")

        # Opcion para imagen subida
        if uploaded_file and uploaded_file.filename:
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

        # Predicción
        reconstructed = model.predict(img)
        print(reconstructed.min(), reconstructed.max())
        reconstructed = np.clip(reconstructed[0], 0.0, 1.0) #sacamos prediccion del batch (1,512,512,3) -> (512,512,3)
        reconstructed_img = (reconstructed * 255).astype("uint8") #conversion de [0,1] -> [0,255]

        # Cálculo de MSE y SSIM
        original_path= os.path.join(app.static_folder, "original", sample_name)
        original_bgr = cv2.imread(original_path) # sacamos imagen original
        original_uint8 = cv2.resize(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), (512, 512)) #pasamos de BGR a RGB
        mse_val  = (calculate_mse(original_uint8, reconstructed_img) / (255.0 ** 2)) 
        ssim_val = calculate_ssim(original_uint8, reconstructed_img)

        # Guardamos el resultado
        output_path = "static/output.png"
        cv2.imwrite(output_path, cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR)) # cv2 espera bgr
        
        return render_template("index.html", images=images, estilo=estilo, mse_val=mse_val, ssim_val=ssim_val, output_path=output_path)
    return render_template("index.html", images=images)

if __name__ == "__main__":
    app.run(debug=True)