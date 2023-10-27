import librosa
import os
import soundfile as sf

def dividir_audio_em_trechos(arquivo_audio, duracao_trecho=5.0, diretorio_saida="trechos"):
    """
    Divide um arquivo de áudio em trechos de duração especificada.

    :param arquivo_audio: Caminho para o arquivo de áudio.
    :param duracao_trecho: Duração de cada trecho em segundos.
    :param diretorio_saida: Diretório onde os trechos serão salvos.
    """
    # Carregar o áudio
    y, sr = librosa.load(arquivo_audio, sr=None)

    # Calcular o número de amostras por trecho
    amostras_por_trecho = int(duracao_trecho * sr)

    # Criar o diretório de saída se ele não existir
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)

    # Dividir o áudio em trechos e salvar cada trecho
    for i, inicio in enumerate(range(0, len(y), amostras_por_trecho)):
        fim = inicio + amostras_por_trecho
        trecho = y[inicio:fim]
        nome_trecho = os.path.join(diretorio_saida, f"2{i}.wav")
        sf.write(nome_trecho, trecho, sr)

# Exemplo de uso
dividir_audio_em_trechos('/home/machine/Downloads/SinaisdeTeste/1.wav',
                        diretorio_saida= "/home/machine/code/pythonProject/testeDeModelo/audios")