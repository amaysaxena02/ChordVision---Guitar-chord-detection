from utils import extract_features, augment_audio, load_audio
from config import SR
import matplotlib.pyplot as plt

# Path to one of your chord audio files
file_path = r"D:\GitHub\ChordVision - Guitar chord detection\Data\Training\Am\Am_acousticguitar_Mari_1.wav"

# --- Load audio first ---
y, sr = load_audio(file_path)
if y is None:
    raise ValueError("Failed to load audio.")

# --- Test feature extraction ---
features = extract_features(y, sr)
print("MFCC feature vector shape:", features.shape)
print("MFCC features (first 5 values):", features[:5])

# --- Test augmentation ---
augmented_samples = augment_audio(y, sr)
print("Number of augmented samples:", len(augmented_samples))

# Optional: plot waveform of original and first augmented audio
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(y, label="Original")
plt.plot(augmented_samples[1], label="Augmented Noise")
plt.legend()
plt.show()
