
# ChordVision - Guitar Chord Detector 

ChordVision is an AI-powered system that automatically detects guitar chords from audio recordings. The system extracts MFCC (Mel-Frequency Cepstral Coefficients) features from audio samples, applies data augmentation for better generalization, and trains a Random Forest classifier to identify chords accurately. The trained model can recognize common beginner and barre chords and can be extended for real-time applications, such as displaying the detected chord in a live video or alongside a chord diagram.

Key features

* Extracts MFCC features from audio recordings.
* Data augmentation with noise, pitch shift, and time-stretch to increase dataset variability.
* Supports 8 beginner chords: Am, F, Em, G, Bb, Bdim, C, Dm
* Trained using a Random Forest classifier achieving ~85% accuracy.
* Modular design for future extensions like live video chord detection.

Technologies & Libraries:

* Python (3.13.3), NumPy, Librosa, scikit-learn, Joblib
* Audio processing and feature extraction
* Random Forest classifier for chord recognition
* Optional integration with Streamlit for interactive UI

## Dataset Used
* https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v2
