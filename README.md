Este repositorio contiene el trabajo de fin de grado de Alba de la Torre, con todo el código y scripts necesarios para entrenar, 
evaluar y probar una red generativa adversarial (GAN) orientada a la reconstrucción digital de obras de arte dañadas. 

Incluye además una interfaz web para facilitar su uso, un generador de daños sintéticos basado en ruido Perlin y scripts de análisis estadístico 
para evaluar el rendimiento del modelo.

<img width="826" alt="Captura de pantalla 2025-06-19 a las 21 54 48" src="https://github.com/user-attachments/assets/37444788-718f-4363-b825-09c53bde8f0d" />
<img width="826" alt="Captura de pantalla 2025-06-19 a las 21 55 20" src="https://github.com/user-attachments/assets/e192d6bf-4983-4ffb-9fea-d8751795df10" />
<img width="826" alt="Captura de pantalla 2025-06-19 a las 21 56 52" src="https://github.com/user-attachments/assets/2b5ee583-8044-48c8-a537-0fc84e68d43c" />


**📂 Contenido del repositorio**
GAN.py
/interfaz
cohenswilcoxon.py
create-mask.py

**🚀 Cómo ejecutar**

Para entrenar el modelo GAN:

```bash
python GAN.py
```

Para generar dataset con imagenes dañadas:

```bash
python create-mask.py --input_root ruta/a/dataset --output_root ruta/a/carpeta_salida
```

Para poner en marcha la interfaz:

```bash
python interface.py
```

**📊 Scripts de evaluación incluidos**
- Wilcoxon signed-rank test: evalúa diferencias significativas entre imágenes originales y reconstruidas.
- Cohen’s d: estima el tamaño del efecto de la reconstrucción sobre la imagen dañada.

