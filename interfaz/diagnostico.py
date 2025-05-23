from keras.models import load_model

# 1. ¿Se carga?
try:
    model = load_model("gan_abstracto.h5", compile=False)
except Exception as e:
    print("NO se carga:", e)          # → causa 1 ó 3
    raise

# 2. ¿La arquitectura es la esperada?
model.summary()                       # compara con gan_impresionistas.h5

# 3. ¿Hace inferencia?
import numpy as np, cv2
img = cv2.imread("../arte-abstracto-damaged/arte-abstracto-damaged/damaged_abstract_edward-corbett_2870.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512,512)).astype("float32")/255.0
out = model.predict(img[None])[0]     # shape (512,512,3)
print("min/max salida:", out.min(), out.max())