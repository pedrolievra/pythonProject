from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio

# Lista de nomes dos arquivos MP4
arquivos_mp4 = ["C:\\Users\\pedro\\PycharmProjects\\pythonProject\\frequencias\\476hz\\476hz.mp4",
                "C:\\Users\\pedro\\PycharmProjects\\pythonProject\\frequencias\\620hz\\620hz.mp4",
                "C:\\Users\\pedro\\PycharmProjects\\pythonProject\\frequencias\\826hz\\826hz.mp4",
                "C:\\Users\\pedro\\PycharmProjects\\pythonProject\\frequencias\\1292hz\\1292hz.mp4",
                "C:\\Users\\pedro\\PycharmProjects\\pythonProject\\frequencias\\1820hz\\1820hz.mp4",
                "C:\\Users\\pedro\\PycharmProjects\\pythonProject\\frequencias\\2018hz\\2018hz.mp4"]

# Loop através dos arquivos MP4
for mp4_file in arquivos_mp4:
    # Extrai o nome do arquivo sem a extensão
    nome_sem_extensao = mp4_file.split(".mp4")[0]

    # Define o nome do arquivo de áudio de saída
    audio_file = f"{nome_sem_extensao}.mp3"

    # Extrai o áudio do arquivo MP4
    ffmpeg_extract_audio(mp4_file, audio_file)

    print(f"Áudio extraído de {mp4_file} e salvo como {audio_file}")
