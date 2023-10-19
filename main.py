import pyaudio
import wave
import time

# Configurações de gravação
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5  # Duração de cada gravação
OUTPUT_FILENAME = "audio_"


def record_audio(interval):
    p = pyaudio.PyAudio()

    for i in range(interval):
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print(f"Gravando áudio {i + 1}...")
        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print(f"Gravação {i + 1} completa.")

        stream.stop_stream()
        stream.close()

        output_filename = f"{OUTPUT_FILENAME}{i + 1}.wav"
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    p.terminate()


if __name__ == "__main__":
    num_recordings = 100  # Número de gravações a serem feitas
    record_audio(num_recordings)
