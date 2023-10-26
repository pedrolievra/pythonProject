diretoriotestedoteste = '/home/machine/Downloads/SinaisdeTeste/pythonProject/testeDeModelo/audios'
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