import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Diretórios de treinamento e teste para as imagens da Transformada de Fourier
diretorio_treinamento = 'C:\\Users\\pedro\\PycharmProjects\\pythonProject\\train\\'
diretorio_teste = 'C:\\Users\\pedro\\PycharmProjects\\pythonProject\\test\\'

# Função para carregar as imagens da Transformada de Fourier
def carregar_transformada_fourier(diretorio):
    transformadas = []
    for arquivo in os.listdir(diretorio):
        imagem = tf.keras.preprocessing.image.load_img(os.path.join(diretorio, arquivo))
        imagem = tf.keras.preprocessing.image.img_to_array(imagem)
        transformadas.append(imagem)
    return np.array(transformadas)

# Carregar as imagens da Transformada de Fourier de treinamento e teste
transformadas_treinamento = carregar_transformada_fourier(diretorio_treinamento)
transformadas_teste = carregar_transformada_fourier(diretorio_teste)

# Rótulos para os dados de treinamento e teste (por exemplo, rótulos de classe)
rotulos_treinamento = []
rotulos_teste = []

# Redimensionar os dados para corresponder ao formato de entrada da LSTM
timesteps, input_features = transformadas_treinamento.shape[1], transformadas_treinamento.shape[2] * transformadas_treinamento.shape[3]
transformadas_treinamento = transformadas_treinamento.reshape(transformadas_treinamento.shape[0], timesteps, input_features)
transformadas_teste = transformadas_teste.reshape(transformadas_teste.shape[0], timesteps, input_features)

# Definir o modelo LSTM
model.add(LSTM(128, input_shape=(timesteps, input_features), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(transformadas_treinamento, rotulos_treinamento, epochs=10, batch_size=32, validation_data=(transformadas_teste, rotulos_teste))

# Avaliar o modelo
resultado = model.evaluate(transformadas_teste, rotulos_teste)
print("Acurácia no conjunto de teste:", resultado[1])
