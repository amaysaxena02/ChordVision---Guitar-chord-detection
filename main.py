import cv2
import numpy as np
import sounddevice as sd
import threading
from collections import deque
from time import time

from model import load_model
from utils import extract_features
from config import SR, N_MFCC, CHORDS

# --- SETTINGS ---
WINDOW_NAME = "ChordVision Real-Time"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
DIAGRAMS_PATH = "diagrams"
BUFFER_SECONDS = 2
PREDICT_INTERVAL = 0.5
MIN_AUDIO_LEVEL = 0.01

# --- GLOBALS ---
audio_buffer = np.zeros(SR * BUFFER_SECONDS, dtype=np.float32)
buffer_lock = threading.Lock()
history = deque(maxlen=5)
current_chord = "..."

# --- AUDIO STREAM ---
def audio_stream():
    def callback(indata, frames, time_info, status):
        global audio_buffer
        if status:
            print(status)
        with buffer_lock:
            audio_buffer = np.roll(audio_buffer, -frames)
            audio_buffer[-frames:] = indata.flatten()
    with sd.InputStream(channels=1, samplerate=SR, callback=callback):
        while True:
            sd.sleep(1000)

# Start audio thread
threading.Thread(target=audio_stream, daemon=True).start()

# --- LOAD MODEL ---
model = load_model("chord_model.pkl")

# --- VIDEO ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
cv2.namedWindow(WINDOW_NAME)

last_predict = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t = time()
    if t - last_predict >= PREDICT_INTERVAL:
        with buffer_lock:
            chunk = audio_buffer.copy()
        last_predict = t

        if np.max(np.abs(chunk)) > MIN_AUDIO_LEVEL:
            try:
                feats = extract_features(chunk, sr=SR, n_mfcc=N_MFCC)
                feats = feats.reshape(1, -1)
                pred = model.predict(feats)[0]
                history.append(pred)
                current_chord = max(set(history), key=history.count)
                print("Predicted chord:", current_chord)  # <- you'll see output now
            except Exception as e:
                print("Prediction error:", e)

    # Overlay chord
    cv2.putText(frame, f"Chord: {current_chord}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

    # Overlay diagram
    try:
        diag = cv2.imread(f"{DIAGRAMS_PATH}/{current_chord}.png", cv2.IMREAD_UNCHANGED)
        if diag is not None:
            diag = cv2.resize(diag, (200, 200))
            h, w = diag.shape[:2]
            x_off = frame.shape[1] - w - 20
            y_off = 20
            if diag.shape[2] == 4:
                alpha = diag[:, :, 3] / 255.0
                for c in range(3):
                    frame[y_off:y_off+h, x_off:x_off+w, c] = (
                        alpha * diag[:, :, c] + (1 - alpha) * frame[y_off:y_off+h, x_off:x_off+w, c]
                    )
            else:
                frame[y_off:y_off+h, x_off:x_off+w] = diag
    except:
        pass

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
