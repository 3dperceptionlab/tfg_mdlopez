import os
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import pickle


with open('/workspace/diccionario_caracteristicas.pickle', 'rb') as f:
    diccionario_caracteristicas = pickle.load(f)

"""
# Obtener todos los valores de los arrays en una sola lista
todos_los_valores = np.concatenate(list(diccionario_caracteristicas.values()))

# Encontrar el valor máximo absoluto de todos los arrays
max_abs_global = np.max(np.abs(todos_los_valores))

# Normalizar cada array en el diccionario utilizando el valor máximo absoluto global
for clave, valor in diccionario_caracteristicas.items():
    valor_normalizado = (valor + max_abs_global) / (2 * max_abs_global)
    diccionario_caracteristicas[clave] = valor_normalizado



print("Las primeras 10 entradas del diccionario de características:")
contador = 0
for clave, valor in diccionario_caracteristicas.items():
    if contador < 28:
        print(clave, ":", valor) 
        contador += 1
    else:
        break

with open('/workspace/diccionario01.pickle', 'wb') as f:
    pickle.dump(diccionario_caracteristicas, f)





    #Correlation heatmap of the variables gsr_avg	gsr_min	gsr_max	gsr_std	ecg_avg	ecg_min	ecg_max	ecg_std	emg_avg	emg_min	emg_max	emg_std of the samplesPainBin
corr = samplesPainBin[['gsr_avg', 'gsr_min', 'gsr_max', 'gsr_std', 'ecg_avg', 'ecg_min', 'ecg_max', 'ecg_std', 'emg_avg', 'emg_min', 'emg_max', 'emg_std','class_id']].corr()
#Export as pdf

fig = sns.heatmap(corr, xticklabels=corr.columns.values,  yticklabels=corr.columns.values, cmap='RdBu_r', center=0, annot=True)
# reduce the text size of every cell value
for text in fig.texts:
    text.set_fontsize(7)

fig.figure.savefig("/workspace/images/heatmap.pdf", bbox_inches='tight')
"""
samples.loc[index, 'gsr_avg'] = rowdf['gsr'].mean()
samples.loc[index, 'gsr_min'] = rowdf['gsr'].min()
samples.loc[index, 'gsr_max'] = rowdf['gsr'].max()
samples.loc[index, 'gsr_std'] = rowdf['gsr'].std()
samples.loc[index, 'ecg_avg'] = rowdf['ecg'].mean()
samples.loc[index, 'ecg_min'] = rowdf['ecg'].min()


import seaborn as sns
import numpy as np

diccionario = {
    "caracGSR": [maximo_GSR, media_GSR, cruces0_GSR],
    "caracECG": [media, amplitud_R, frecuencia_cardiaca, intervalo_RR, intervalo_ST, intervalo_QP, intervalo_TQ, cruces0, amplitud_T, amplitud_P],
    "caracEMG": [media_EMG, cuspide, varianza, amplitud_EMG, amplitud_EMG_N, cruces0_EMG, densidad]
}

vector = []
for key in diccionario:
    vector.extend(diccionario[key])
# Crear una matriz de correlación ficticia para ilustrar el ejemplo
corr = np.random.rand(len(vector), len(vector))

fig = sns.heatmap(corr, xticklabels=vector, yticklabels=vector, cmap='RdBu_r', center=0, annot=True)
for text in fig.texts:
    text.set_fontsize(7)

# Guardar el gráfico en un archivo PDF
fig.figure.savefig("/workspace/images/heatmap.pdf", bbox_inches='tight')
