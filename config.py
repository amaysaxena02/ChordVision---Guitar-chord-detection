DATASET_PATH = r"D:\GitHub\ChordVision - Guitar chord detection\Data"
TRAINING_FOLDER = r"D:\GitHub\ChordVision - Guitar chord detection\Data\Training"

CHORDS = ["Am", "F", "Em", "G", "Bb", "Bdim", "C", "Dm"]

# Audio processing settings
SR = 22050      # sample rate
N_MFCC = 13     # number of MFCCs features

# Model augmentation settings
AUGMENT = True
MODEL_FILE = "chord_model.pkl"