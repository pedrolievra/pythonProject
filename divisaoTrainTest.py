from pydub import AudioSegment
import os

# Diretório do arquivo MP3 de entrada e das pastas de saída
input_file = "C:\\Users\\pedro\\PycharmProjects\\pythonProject\\frequencias\\2018hz\\2018hz.mp3"
output_train_folder = "C:\\Users\\pedro\\PycharmProjects\\pythonProject\\frequencias\\2018hz\\train"
output_test_folder = "C:\\Users\\pedro\\PycharmProjects\\pythonProject\\frequencias\\2018hz\\test"

# Verifica se as pastas de saída existem, senão cria
os.makedirs(output_train_folder, exist_ok=True)
os.makedirs(output_test_folder, exist_ok=True)

# Carrega o arquivo MP3
audio = AudioSegment.from_mp3(input_file)

# Duração dos áudios de saída (5 segundos)
segment_duration = 5000  # em milissegundos

# Quantidade total de áudios de saída
total_segments = len(audio) // segment_duration

# Quantidade de áudios para o conjunto de treinamento e teste
train_count = int(total_segments * 0.85)
test_count = total_segments - train_count

# Divide o áudio em pequenos segmentos
for i in range(total_segments):
    segment = audio[i * segment_duration:(i + 1) * segment_duration]

    # Escolhe a pasta de saída com base na divisão desejada
    if i < train_count:
        output_folder = output_train_folder
    else:
        output_folder = output_test_folder

    # Salva o segmento de áudio no diretório apropriado
    output_file = os.path.join(output_folder, f"segment_{i}.mp3")
    segment.export(output_file, format="mp3")

print("Concluído! O arquivo MP3 foi dividido em segmentos de 5 segundos.")
