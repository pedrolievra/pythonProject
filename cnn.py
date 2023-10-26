import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Diretórios base para 2 e 3
diretorio_5kHz = 'C:\\Users\\pedro\\PycharmProjects\\pythonProject\\2\\'
diretorio_10kHz = 'C:\\Users\\pedro\\PycharmProjects\\pythonProject\\3\\'

def carregar_transformada_fourier_e_rotulos(diretorio, rotulo):
    transformadas = []
    rotulos = []

    for arquivo in os.listdir(diretorio):
        rotulos.append(rotulo)
        imagem = tf.keras.preprocessing.image.load_img(os.path.join(diretorio, arquivo))
        imagem = tf.keras.preprocessing.image.img_to_array(imagem)
        transformadas.append(imagem)

    return np.array(transformadas), np.array(rotulos)

# Carregar e rotular os espectrogramas de treinamento
transformadas_5kHz_treino, rotulos_5kHz_treino = carregar_transformada_fourier_e_rotulos(os.path.join(diretorio_5kHz, 'train'), 0)
transformadas_10kHz_treino, rotulos_10kHz_treino = carregar_transformada_fourier_e_rotulos(os.path.join(diretorio_10kHz, 'train'), 1)

# Carregar e rotular os espectrogramas de teste
transformadas_5kHz_teste, rotulos_5kHz_teste = carregar_transformada_fourier_e_rotulos(os.path.join(diretorio_5kHz, 'test'), 0)
transformadas_10kHz_teste, rotulos_10kHz_teste = carregar_transformada_fourier_e_rotulos(os.path.join(diretorio_10kHz, 'test'), 1)

# Concatenar os dados e rotular em formatos categóricos one-hot
transformadas_treinamento = np.concatenate([transformadas_5kHz_treino, transformadas_10kHz_treino], axis=0)
rotulos_treinamento = tf.keras.utils.to_categorical(np.concatenate([rotulos_5kHz_treino, rotulos_10kHz_treino], axis=0), 2)

transformadas_teste = np.concatenate([transformadas_5kHz_teste, transformadas_10kHz_teste], axis=0)
rotulos_teste = tf.keras.utils.to_categorical(np.concatenate([rotulos_5kHz_teste, rotulos_10kHz_teste], axis=0), 2)

# Definir o modelo CNN
input_shape = transformadas_treinamento[0].shape

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # 2 classes

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(transformadas_treinamento, rotulos_treinamento, epochs=10, batch_size=32, validation_data=(transformadas_teste, rotulos_teste))

# Avaliar o modelo
resultado = model.evaluate(transformadas_teste, rotulos_teste)
print("Acurácia no conjunto de teste:", resultado[1])
