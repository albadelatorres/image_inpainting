#!/usr/bin/env python3
"""
Script de python para calcular tamaño del efecto entre GAN e ICT
Acepta dos ficheros csv con columnas:
  Imagen | MSE | SSIM | ORB | FSIM | LPIPS | DISTS
Uso: python cohenswilcoxon.py
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

# Definimos constantes 
GAN_FILE  = "metrics_GAN.csv"
ICT_FILE  = "metrics_ICT.csv"
METRICS   = ["LPIPS", "DISTS", "FSIM", "SSIM", "MSE", "ORB"]
# Indica cuál es "mejor cuanto menor" para fijar el signo
LOWER_IS_BETTER = {"LPIPS", "DISTS", "MSE"}     # el resto: mayor = mejor

gan = pd.read_csv(GAN_FILE, sep=";")
ict = pd.read_csv(ICT_FILE, sep=";")

gan.head()
ict.head()

# Combinar por imagen
# Esto nos deja un dataframe con la forma
#     Imagen  LPIPS_GAN  SSIM_GAN   ...  LPIPS_ICT  SSIM_ICT   ...
#0  img1.jpg      0.205      0.82           0.150      0.79
#1  img2.jpg      0.178      0.84           0.140      0.83
#2  img3.jpg      0.241      0.80           0.162      0.76
merged = gan.merge(ict, on="Imagen", suffixes=("_GAN", "_ICT"))

rows = []
for m in METRICS:
    
    # Cogemos las métricas m _GAN _ICT
    x_gan = merged[f"{m}_GAN"]
    x_ict = merged[f"{m}_ICT"]

    # Diferenciamos por signo: positivo = ICT mejor
    if m in LOWER_IS_BETTER:
        diff = x_gan - x_ict          # GAN – ICT  (menor = mejor)
    else:
        diff = x_ict - x_gan          # ICT – GAN  (mayor = mejor)

    # Calculamos delta
    delta   = diff.mean()
    
    # Calculamos d de Cohen
    d       = delta / diff.std(ddof=1)

    # Calculamos tests (devuelve de dos colas por defecto)
    t_p     = ttest_rel(x_gan, x_ict).pvalue
    
    # Convertimos a t student de una cola dependiendo del signo de delta
    if delta > 0:
        t_p_one = t_p / 2 # Si ICT es mejor, solo dividimos entre 2
    else:
        t_p_one = 1 - t_p / 2 # Si GAN es mejor, restamos 1 - t_p para moverlo a la cola "correcta"

    # Calculamos Wilcoxon
    if m in LOWER_IS_BETTER:
        W, p_w = wilcoxon(x_gan, x_ict)
    else:
        W, p_w = wilcoxon(x_ict, x_gan)
    N       = len(diff)
    z       = (W - N*(N+1)/4) / np.sqrt(N*(N+1)*(2*N+1)/24)
    
    # Calculamos r de Wilcoxon
    r       = abs(z) / np.sqrt(N)
    
    # Convertimos a una cola
    if delta > 0:
        p_w_one = p_w / 2
    else:
        p_w_one = 1 - p_w / 2
    rows.append([m, delta, d, t_p_one, r, p_w_one])

# Mostrar y guardar
results = pd.DataFrame(rows, columns=["Métrica", "Δ", "Cohen d", "p-t_one", "r Wilcoxon", "p-W_one"])
print(results.to_string(index=False, float_format="%.4f"))
results.to_csv("effect_sizes.csv", index=False)