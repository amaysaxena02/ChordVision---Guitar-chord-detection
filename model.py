import joblib
from sklearn.ensemble import RandomForestClassifier
from utils import extract_features

class ChordModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)  # X should be 2D: (1, N)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

def load_model(path="chord_model.pkl"):
    model = ChordModel()
    model.load(path)
    return model

def predict_chord(model, file_path_or_array):
    features = extract_features(file_path_or_array)
    features = features.reshape(1, -1)  # Ensure 2D for sklearn
    return model.predict(features)
