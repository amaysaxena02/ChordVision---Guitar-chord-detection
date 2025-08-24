import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import TRAINING_FOLDER, CHORDS, SR, N_MFCC, AUGMENT, MODEL_FILE
from utils import extract_features, augment_audio, load_audio
from model import ChordModel

def prepare_dataset():
    X, y = [], []

    for chord in CHORDS:
        chord_folder = os.path.join(TRAINING_FOLDER, chord)
        if not os.path.exists(chord_folder):
            print(f"Chord folder not found: {chord_folder}")
            continue

        for file in os.listdir(chord_folder):
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(chord_folder, file)

            # Load audio
            audio, sr = load_audio(file_path)
            if audio is None:
                continue

            # Extract features
            features = extract_features(audio)
            X.append(features)
            y.append(chord)

            # Augmentation
            if AUGMENT:
                augmented_samples = augment_audio(audio, sr)
                for aug in augmented_samples:
                    features_aug = extract_features(aug)
                    X.append(features_aug)
                    y.append(chord)

    return np.array(X), np.array(y)

def train_model():
    print("Preparing dataset...")
    X, y = prepare_dataset()
    print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features")

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    model = ChordModel()
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = [model.predict(x.reshape(1, -1)) for x in X_test]
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")

    print(f"Saving model to {MODEL_FILE}...")
    model.save(MODEL_FILE)
    print("Training complete.")

if __name__ == "__main__":
    train_model()
