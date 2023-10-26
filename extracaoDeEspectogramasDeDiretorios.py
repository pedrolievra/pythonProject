import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def extract_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def process_directory(dirs, base_save_dir):
    all_spectrograms = []
    for directory in dirs:
        freq_type = os.path.basename(directory)
        for filename in os.listdir(directory):
            if filename.endswith('.wav') or filename.endswith('.mp3'):
                audio_path = os.path.join(directory, filename)

                # Define the save directory based on train or test and frequency type
                train_test_split_dir = "train" if np.random.rand() < 0.8 else "test"
                save_directory = os.path.join(base_save_dir, train_test_split_dir, freq_type)

                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                save_path = os.path.join(save_directory, filename + ".png")
                extract_spectrogram(audio_path, save_path)
                all_spectrograms.append(save_path)
    return all_spectrograms


# Uso:

dirs = ['C:\\Users\\pedro\\PycharmProjects\\pythonProject\\frequenciasDefinitivas\\1',
        'C:\\Users\\pedro\\PycharmProjects\pythonProject\\frequenciasDefinitivas\\2',
        'C:\\Users\\pedro\\PycharmProjects\\pythonProject\\frequenciasDefinitivas\\3',
        'C:\\Users\\pedro\\PycharmProjects\pythonProject\\frequenciasDefinitivas\\4']

base_save_dir = "C:\\Users\\pedro\\PycharmProjects\\pythonProject\\testeDeModelo"

all_spectrograms = process_directory(dirs, base_save_dir)
train_data = [path for path in all_spectrograms if "\\train\\" in path]
test_data = [path for path in all_spectrograms if "\\test\\" in path]

print("Train Data:", len(train_data))
print("Test Data:", len(test_data))
