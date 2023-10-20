import os
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# Diretório contendo os arquivos WAV
diretorio_audio = '/home/machine/Downloads/SinaisdeTeste/pythonProject/frequencias/620hz/test'

# Diretório de saída para os espectrogramas de treinamento e teste
diretorio_treinamento = '/home/machine/Downloads/SinaisdeTeste/pythonProject/frequencias/476hz/test'
diretorio_teste = '/home/machine/Downloads/SinaisdeTeste/pythonProject/frequencias/476hz/train'

# Listar os arquivos WAV no diretório
arquivos_wav = [f for f in os.listdir(diretorio_audio) if f.endswith('.wav')]

# Dividir os dados em treinamento (85%) e teste (15%)
treinamento, teste = train_test_split(arquivos_wav, test_size=0.15, random_state=42)

# Função para criar e salvar o espectrograma
def criar_espectrograma(arquivo, diretorio_saida):
    audio, _ = librosa.load(os.path.join(diretorio_audio, arquivo))
    espectrograma = np.abs(np.fft.fft(audio))
    plt.figure(figsize=(10, 4))
    plt.specgram(audio, NFFT=2048, Fs=2, Fc=0, noverlap=1024, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma')
    plt.savefig(os.path.join(diretorio_saida, os.path.splitext(arquivo)[0] + '.png'))
    plt.close()

# Criar espectrogramas para os dados de treinamento
for arquivo in treinamento:
    criar_espectrograma(arquivo, diretorio_treinamento)

# Criar espectrogramas para os dados de teste
for arquivo in teste:
    criar_espectrograma(arquivo, diretorio_teste)

print("Espectrogramas criados e divididos em treinamento e teste.")