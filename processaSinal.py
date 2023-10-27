from scipy.signal import butter, lfilter
import soundfile as sf
import matplotlib.pyplot as plt

# Carregue o arquivo de áudio
audio_file = "/home/machine/Downloads/SinaisdeTeste/1.wav"
y, sr = sf.read(audio_file)

# Especificações do filtro Butterworth (exemplo: passa-faixa de 4.5 kHz a 5.5 kHz)
frequencia_min = 4500
frequencia_max = 5500
ordem = 2  # Ajuste a ordem do filtro conforme necessário

# Projete o filtro passa-faixa Butterworth
b, a = butter(ordem, [frequencia_min / (0.5 * sr), frequencia_max / (0.5 * sr)], btype='band')

# Aplique o filtro ao sinal de áudio
sinal_filtrado = lfilter(b, a, y)

# Plote o sinal de áudio original
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(y, color = "orange")
plt.title("Sinal de Áudio Original")

# Plote o sinal de áudio filtrado
plt.subplot(2, 1, 2)
plt.plot(sinal_filtrado, color = "blue")
plt.title("Sinal de Áudio Filtrado")

plt.tight_layout()
plt.show()
# Especifique o caminho para o arquivo de saída (novo arquivo de áudio filtrado)
output_file = "audio_filtrado.wav"  # Nome do arquivo de saída

# Exporte o áudio filtrado para o arquivo de saída
sf.write(output_file, sinal_filtrado, sr)
