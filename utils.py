import librosa
import numpy as np
from config import SR, N_MFCC

def extract_features(input_data, sr=SR, n_mfcc=N_MFCC):
    """
    input_data: either file path (str) or raw audio numpy array
    Returns: 1D MFCC feature vector
    """
    # Load audio if path
    if isinstance(input_data, str):
        y, _ = librosa.load(input_data, sr=sr, duration=3)
    else:
        y = np.asarray(input_data).flatten()  # ensure 1D

    # Trim or pad audio to at least 1 second
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)), mode='constant')

    # Extract MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    return np.mean(mfccs.T, axis=0)

def augment_audio(y, sr=SR):
    """
    Generate augmented versions of a raw audio array.
    Returns list of 1D numpy arrays.
    """
    augmented_samples = []

    # Original
    augmented_samples.append(y)

    # Noise
    y_noise = y + 0.005 * np.random.randn(len(y))
    augmented_samples.append(y_noise)

    # Time stretch
    y_stretch = librosa.effects.time_stretch(y, rate=1.1)
    augmented_samples.append(y_stretch.flatten())  # ensure 1D

    # Pitch shift
    y_shift = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=1)
    augmented_samples.append(y_shift.flatten())  # ensure 1D

    return augmented_samples

def load_audio(file_path, sr=SR, duration=3):
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        return np.asarray(y).flatten(), sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None
