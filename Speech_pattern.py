import librosa
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)

        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr = sr,n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio,sr=sr).T, axis=0)
        return np.concatenate((mfcc, chroma, spectral_contrast))

    except Exception as e :
        print(f"Error processing {file_path}: {e}")
        return None



def predict_speech_tone(audio_path):
    feature = extract_features(audio_path)
    if feature is not None:
        prediction = model.predict([feature])
        if prediction[0] == 0:
            return "Neutral"
        elif prediction[0] == 1:
            return "Happy"
        elif prediction[0] == 2:
            return "Sad"
        # Add more tone labels as needed
    else:
        return "Error in processing audio file."

result = predict_speech_tone('audio4.wav')
print("Predicted Speech Tone:", result)
def load_data(file_path, labels):
    features = []
    valid_labels = []
    for i, file_path in enumerate(file_path):
        feature = extract_features(file_path)
        if feature is not None:
            features.append(feature)
            valid_labels.append(labels[i])

    return np.array(features),np.array(valid_labels)

audio_files =  ['audio1.wav', 'audio2.wav', 'audio3.wav']
labels = [0,1,0]

X, y = load_data(audio_files, labels)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Clasification Report:\n",classification_report(y_test,y_pred))


def predict_mental_health(audio_path):
    feature = extract_features(audio_path)
    if feature is not None:
        prediction = model.predict([feature])
        return "Mental Health Issue" if prediction[0] == 1 else "No Issue"
    else:
        return "Error in processing audio file."

result = predict_mental_health('audio4.wav')
print("Prediction:",result)
