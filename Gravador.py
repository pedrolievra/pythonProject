import sounddevice as sd
import scipy.io.wavfile as wav
import time
import os

# Diretório onde os arquivos de áudio serão salvos
output_directory = 'audios/'

# Crie o diretório se ele não existir
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Duração total da gravação em segundos
duration = 90  # 1 minuto e meio

# Intervalo de gravação em segundos
interval = 5

# Número total de segmentos a serem gravados
num_segments = int(duration / interval)

for i in range(num_segments):
    # Grave áudio
    audio_data = sd.rec(int(interval * 44100), samplerate=44100, channels=2)
    sd.wait()
    
    # Nome do arquivo de áudio com base no timestamp
    timestamp = int(time.time())
    audio_filename = f'{output_directory}audio_{timestamp}.wav'
    
    # Salve o arquivo de áudio
    wav.write(audio_filename, 44100, audio_data)

    print(f'Gravado {audio_filename}')
    time.sleep(interval)

print("Gravação concluída.")
