import os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

# Diretórios de treinamento e teste para os espectrogramas
diretorio_treinamento = '/home/machine/Downloads/SinaisdeTeste/pythonProject/train'
diretorio_teste = '/home/machine/Downloads/SinaisdeTeste/pythonProject/test'

# Função para carregar os espectrogramas
def carregar_transformada_fourier_e_rotulos(diretorio):
    transformadas = []
    rotulos = []

    for idx, subdiretorio in enumerate(os.listdir(diretorio)):
        subdiretorio_path = os.path.join(diretorio, subdiretorio)

        # Verifique se o item é um diretório antes de tentar listar seus arquivos
        if os.path.isdir(subdiretorio_path):
            for arquivo in os.listdir(subdiretorio_path):
                # O índice do subdiretório (frequência) é usado como rótulo
                rotulos.append(idx)

                imagem = tf.keras.preprocessing.image.load_img(os.path.join(subdiretorio_path, arquivo))
                imagem = tf.keras.preprocessing.image.img_to_array(imagem)
                transformadas.append(imagem)

    return np.array(transformadas), np.array(rotulos)

# Carregando e rotulando os espectrogramas
transformadas_treinamento, rotulos_treinamento = carregar_transformada_fourier_e_rotulos(diretorio_treinamento)
transformadas_teste, rotulos_teste = carregar_transformada_fourier_e_rotulos(diretorio_teste)

# Remodelar os dados para terem duas dimensões: número de amostras e número de características
transformadas_treinamento = transformadas_treinamento.reshape(transformadas_treinamento.shape[0], -1)
transformadas_teste = transformadas_teste.reshape(transformadas_teste.shape[0], -1)

# Definir o modelo SVM
clf = svm.SVC(kernel='linear')  # Usando um kernel linear para a SVM

# Treinar o modelo
clf.fit(transformadas_treinamento, rotulos_treinamento)

# Fazer previsões no conjunto de teste
rotulos_preditos = clf.predict(transformadas_teste)

# Fazer previsões no conjunto de treinamento
rotulos_preditos_treinamento = clf.predict(transformadas_treinamento)

# Avaliar o modelo no conjunto de treinamento
acuracia_treinamento = accuracy_score(rotulos_treinamento, rotulos_preditos_treinamento)

# Avaliar o modelo
acuracia = accuracy_score(rotulos_teste, rotulos_preditos)
f1 = f1_score(rotulos_teste, rotulos_preditos, average='weighted')  # Usando média ponderada para classes desbalanceadas
precisao = precision_score(rotulos_teste, rotulos_preditos, average='weighted')  # Usando média ponderada

print("Acurácia no conjunto de treinamento:", acuracia_treinamento)
print("Acurácia no conjunto de teste:", acuracia)
print("F1-score no conjunto de teste:", f1)
print("Precisão no conjunto de teste:", precisao)

diretoriotestedoteste = '/home/machine/Downloads/SinaisdeTeste/pythonProject/testeDeModelo/espectogramas'
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

# Carregar os espectrogramas diretamente do diretório
espectrogramas_capturados = carregar_transformada_fourier_direto(diretoriotestedoteste)

# Verificar se os espectrogramas foram carregados corretamente
if espectrogramas_capturados.size == 0:
    print("Nenhum espectrograma foi carregado. Verifique o diretório e os arquivos.")
    exit()

espectrogramas_capturados = espectrogramas_capturados.reshape(espectrogramas_capturados.shape[0], -1)

# Classificar os espectrogramas capturados com a SVM
rotulos_preditos_capturados = clf.predict(espectrogramas_capturados)

# Mapear os índices de classes de volta para as frequências
frequencias = ["freq1", "freq2", "freq3", "freq4", "freq5", "freq6"]
frequencias_preditas_capturados = [frequencias[idx] for idx in rotulos_preditos_capturados]

print("Frequências detectadas nos espectrogramas capturados:", frequencias_preditas_capturados)
