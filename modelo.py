import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Diretórios de treinamento e teste para os espectrogramas
diretorio_treinamento = 'C:\\Users\\pedro\\PycharmProjects\\pythonProject\\train\\'
diretorio_teste = 'C:\\Users\\pedro\\PycharmProjects\\pythonProject\\test\\'

# Função para carregar os espectrogramas
def carregar_transformada_fourier_e_rotulos(diretorio):
    transformadas = []
    rotulos = []

    for idx, subdiretorio in enumerate(os.listdir(diretorio)):
        subdiretorio_path = os.path.join(diretorio, subdiretorio)

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

rotulos_treinamento = tf.keras.utils.to_categorical(rotulos_treinamento, 6)
rotulos_teste = tf.keras.utils.to_categorical(rotulos_teste, 6)

timesteps = transformadas_treinamento.shape[1]
features = transformadas_treinamento.shape[2] * transformadas_treinamento.shape[3]
transformadas_treinamento = transformadas_treinamento.reshape(-1, timesteps, features)
transformadas_teste = transformadas_teste.reshape(-1, timesteps, features)


# Definir o modelo LSTM
model = Sequential()
model.add(LSTM(128, input_shape=(timesteps, features), return_sequences=True))
model.add(LSTM(128))  # Aqui, removemos o 'return_sequences=True'
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))  # 6 classes


# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(transformadas_treinamento, rotulos_treinamento, epochs=10, batch_size=32, validation_data=(transformadas_teste, rotulos_teste))

# Avaliar o modelo
resultado = model.evaluate(transformadas_teste, rotulos_teste)
print("Acurácia no conjunto de teste:", resultado[1])
