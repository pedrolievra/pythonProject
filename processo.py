import os
import time
import scipy.io.wavfile as wav
import sounddevice as sd
from scipy.signal import butter, lfilter
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from sklearn import svm
import pickle

# Configurações
output_directory = '/home/machine/code/pythonProject/processo/audios'
pasta_saida = '/home/machine/code/pythonProject/processo/audios/espectogramas'
duration = 90  # 1 minuto e 30 segundos
interval = 5
nome_arquivo_modelo = '/home/machine/code/pythonProject/modelo_svm.sav'
frequencias = ["1kHz", "5khz", "10kHz", "500Hz"]

def gravar_audio_completo():
    print("Iniciando a gravação do áudio...")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=2)
    sd.wait()
    timestamp = int(time.time())
    audio_filename = f'{output_directory}audio_completo_{timestamp}.wav'
    wav.write(audio_filename, 44100, audio_data)
    print(f'Gravado {audio_filename}')
    return audio_filename
    print("Gravação do áudio concluída.")
    

def aplicar_filtro(audio_path):
    print("Iniciando a aplicação do filtro...")
    y, sr = sf.read(audio_path)
    frequencia_min = 4500
    frequencia_max = 5500
    ordem = 2
    b, a = butter(ordem, [frequencia_min / (0.5 * sr), frequencia_max / (0.5 * sr)], btype='band')
    sinal_filtrado = lfilter(b, a, y)
    output_file = os.path.join(output_directory, "filtered_complete_audio.wav")
    sf.write(output_file, sinal_filtrado, sr)
    return output_file
    print("Aplicação do filtro concluída.")

def dividir_audio_em_segmentos(audio_path):
    print("Iniciando a divisão do áudio em segmentos...")
    y, sr = sf.read(audio_path)
    num_segments = int(duration / interval)
    segment_files = []
    for i in range(num_segments):
        start_sample = i * interval * sr
        end_sample = (i + 1) * interval * sr
        segment = y[start_sample:end_sample]
        segment_filename = f'{output_directory}segment_{i}.wav'
        sf.write(segment_filename, segment, sr)
        segment_files.append(segment_filename)
    return segment_files
    print("Divisão do áudio em segmentos concluída.")

def gerar_espectrogramas(segment_files):
    print("Iniciando a geração dos espectrogramas...")
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    for segment_file in segment_files:
        y, sr = librosa.load(segment_file)
        espectrograma = librosa.feature.melspectrogram(y=y, sr=sr)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(espectrograma, ref=np.max), y_axis='mel', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Espectrograma')
        plt.tight_layout()
        nome_saida = os.path.splitext(os.path.basename(segment_file))[0] + '_espectrograma.png'
        caminho_saida = os.path.join(pasta_saida, nome_saida)
        plt.savefig(caminho_saida)
        plt.close()
    print('Espectrogramas gerados e salvos com sucesso!')
    print("Geração dos espectrogramas concluída.")

def carregar_transformada_fourier_direto(diretorio):
    transformadas = []
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith('.png'):
            imagem_path = os.path.join(diretorio, arquivo)
            try:
                imagem = tf.keras.preprocessing.image.load_img(imagem_path)
                imagem = tf.keras.preprocessing.image.img_to_array(imagem)
                transformadas.append(imagem)
            except Exception as e:
                print(f"Erro ao carregar o arquivo {arquivo}: {e}")
    return np.array(transformadas)

def classificar_espectrogramas():
    print("Iniciando a classificação dos espectrogramas...")
    clf = pickle.load(open(nome_arquivo_modelo, 'rb'))
    espectrogramas_capturados = carregar_transformada_fourier_direto(pasta_saida)
    if espectrogramas_capturados.size == 0:
        print("Nenhum espectrograma foi carregado. Verifique o diretório e os arquivos.")
        exit()
    espectrogramas_capturados = espectrogramas_capturados.reshape(espectrogramas_capturados.shape[0], -1)
    rotulos_preditos_capturados = clf.predict(espectrogramas_capturados)
    frequencias_preditas_capturados = [frequencias[idx] for idx in rotulos_preditos_capturados]
    print("Frequências detectadas nos espectrogramas capturados:", frequencias_preditas_capturados)
    print("Classificação dos espectrogramas concluída.")

if __name__ == "__main__":
    audio_filename = gravar_audio_completo()
    filtered_audio = aplicar_filtro(audio_filename)
    segment_files = dividir_audio_em_segmentos(filtered_audio)
    gerar_espectrogramas(segment_files)
    classificar_espectrogramas()
