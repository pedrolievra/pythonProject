import os
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

# Defina a pasta de entrada e saída
pasta_entrada = '/home/machine/code/pythonProject/testeDeModelo/audios'
pasta_saida = '/home/machine/code/pythonProject/processo/audios/espectogramas'

# Crie a pasta de saída se ela não existir
if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

# Lista todos os arquivos WAV na pasta de entrada
arquivos_wav = [f for f in os.listdir(pasta_entrada) if f.endswith('.wav')]

# Loop pelos arquivos WAV e crie os espectrogramas
for arquivo_wav in arquivos_wav:
    arquivo_entrada = os.path.join(pasta_entrada, arquivo_wav)

    # Carregue o áudio usando librosa
    y, sr = librosa.load(arquivo_entrada)

    # Crie o espectrograma
    espectrograma = librosa.feature.melspectrogram(y=y, sr=sr)

    # Plote e salve o espectrograma
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(espectrograma, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma')
    plt.tight_layout()

    nome_saida = os.path.splitext(arquivo_wav)[0] + '_espectrograma.png'
    caminho_saida = os.path.join(pasta_saida, nome_saida)
    plt.savefig(caminho_saida)
    plt.close()

print('Espectrogramas gerados e salvos com sucesso!')