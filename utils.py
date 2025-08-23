import librosa
import numpy as np
from config import SR, N_MFCC

def extract_features(y, sr=SR, n_mfcc=N_MFCC):
    """
    Extracts MFCC features from a loaded audio signal.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def augment_audio(y, sr=SR):
    """
    Generates augmented versions of the audio.
    """
    augmented_samples = []

    augmented_samples.append(y)  # original
    augmented_samples.append(y + 0.005 * np.random.randn(len(y)))  # noise
    augmented_samples.append(librosa.effects.time_stretch(y, rate=1.1))  # time stretch
    augmented_samples.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=1))  # pitch shift

    return augmented_samples

def load_audio(file_path, sr=SR, duration=3):
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None
