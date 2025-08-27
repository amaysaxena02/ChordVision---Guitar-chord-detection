# 🎸 ChordVision — Guitar Chord Detection

ChordVision is a **Python-based guitar chord detection system** that identifies chords from live microphone input or audio files. It helps guitarists visualize the chords they are playing in real time and supports practice, learning, and analysis.

---

## 🔧 Quick Start

### 1) Create & activate a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

**Key dependencies include:**
- `numpy` — numerical computations  
- `scipy` — signal processing  
- `librosa` — audio feature extraction  
- `sounddevice` — real-time audio input from microphone  
- `matplotlib` — visualization of chords and waveforms  
- `torch` — if using any deep learning models for chord classification

---

### 3) Run the app
```bash
python main.py
```

- The app supports **live microphone input** and **pre-recorded audio files** for chord detection.
- Visualizes detected chords on a **timeline** and displays the chord name in real-time.

---

## 🎛️ Features

- Detects major, minor, and barre chords from guitar audio.  
- Supports **live detection** from your guitar using a microphone.  
- Visualizes chord progression with plots using `matplotlib`.  
- Optionally uses **machine learning models** for improved detection accuracy.  
- Works locally without the need for an internet connection.

---

## ❓ Usage Tips

- Ensure your microphone is working and has low background noise for accurate detection.  
- Play slowly and clearly for better results, especially with complex chords.  
- For pre-recorded audio, ensure the sample rate is compatible (usually 44.1kHz).  
- Adjust `hop_length` and `window_size` in the config for optimal real-time performance.

---

## 🧱 What this app is (and isn’t)

- ✅ Local guitar chord detection system using Python.  
- ✅ Visualizes chords in real-time and supports audio file input.  
- ❌ Not a full DAW or music production software.  
- ❌ Accuracy may vary with noisy input or non-standard tunings.

---

## 📄 License

This project is provided "as is". Please ensure compliance with the licenses of the used libraries (`librosa`, `torch`, etc.).

