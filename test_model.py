# test_model.py
from model import load_model, predict_chord

# Path to a sample audio file
file_path = r"D:\GitHub\ChordVision - Guitar chord detection\Data\Training\Am\Am_acousticguitar_Mari_1.wav"

# Load the trained model
model = load_model()  # automatically loads chord_model.pkl

# Predict the chord
predicted_chord = predict_chord(model, file_path)

print("Predicted chord:", predicted_chord)
