from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

autoencoder_abstracto = load_model('gan_abstracto.keras')
dam_path = '../arte-abstracto-damaged/arte-abstracto-damaged/damaged_abstract_edward-corbett_2870.jpg'
or_path = '../arte-abstracto-damaged/arte-abstracto-damaged/damaged_abstract_edward-corbett_2870.jpg'
dam_image = cv2.imread(dam_path)  
or_image = cv2.imread(or_path)
img = cv2.resize(dam_image, (512, 512))
img = img.astype("float32") / 255.0
inp = np.expand_dims(img, 0) 

reconstructed = autoencoder_abstracto.predict(inp)[0]

plt.figure(figsize=(8,4))
plt.subplot(1,2,2); plt.title("Reconstruida"); plt.imshow(reconstructed); plt.axis("off")
plt.show()