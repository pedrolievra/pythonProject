import os
import numpy as np
import tensorflow as tf
from sklearn import svm
import pickle

# Diretório onde os espectrogramas estão localizados para teste
diretoriotestedoteste = '/home/machine/code/pythonProject/testeDeModelo/espectogramas'

# Carregar e remodelar os espectrogramas capturados
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

# Carregar o modelo treinado
nome_arquivo_modelo = 'modelo_svm.sav'
clf = pickle.load(open(nome_arquivo_modelo, 'rb'))

# Carregar os espectrogramas diretamente do diretório
espectrogramas_capturados = carregar_transformada_fourier_direto(diretoriotestedoteste)

# Verificar se os espectrogramas foram carregados corretamente
if espectrogramas_capturados.size == 0:
    print("Nenhum espectrograma foi carregado. Verifique o diretório e os arquivos.")
    exit()

espectrogramas_capturados = espectrogramas_capturados.reshape(espectrogramas_capturados.shape[0], -1)

# Classificar os espectrogramas capturados com a SVM
rotulos_preditos_capturados = clf.predict(espectrogramas_capturados)

frequencias = ["1kHz", "5khz", "10kHz", "500Hz"]
frequencias_preditas_capturados = [frequencias[idx] for idx in rotulos_preditos_capturados]

print("Frequências detectadas nos espectrogramas capturados:", frequencias_preditas_capturados)
