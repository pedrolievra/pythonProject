import librosa
import numpy as np
import soundfile as sf

PATH = "C:\\Users\\pedro\\PycharmProjects\\pythonProject\\frequenciasDefinitivas\\10kHz\\10000.wav" # Replace with your file here
original_audio, sample_rate = librosa.load(PATH)
