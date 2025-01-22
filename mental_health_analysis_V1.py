''' In this version of my project i have done it without emotion detection '''
import re
import nltk
import librosa
import numpy as np
import pandas as pd
import cv2
import json
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from keras.models import load_model

# Download necessary NLTK resources (only needed once)
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Define mental health-related keywords and solutions
MENTAL_HEALTH_KEYWORDS = [
    "depression", "anxiety", "stress", "suicide", "hopeless", "panic",
    "fear", "sadness", "lonely", "isolation", "overwhelmed", "worthless",
    "self-harm", "insomnia", "fatigue", "anger", "guilt"
]

MENTAL_HEALTH_SOLUTIONS = {
    "depression": "Consider reaching out to a mental health professional and maintaining a daily journal to track your emotions.",
    "anxiety": "Practice mindfulness and deep breathing exercises. Regular physical activity can also help.",
    "stress": "Try managing your time effectively, prioritize tasks, and engage in relaxing activities such as yoga or meditation.",
    "suicide": "Please seek immediate help from a professional or contact a crisis hotline. You are not alone.",
    "hopeless": "Connect with supportive friends or family. Sometimes talking about your feelings can help.",
    "panic": "Try grounding techniques, such as focusing on your breathing or describing your surroundings.",
    "fear": "Identify the source of your fear and take small steps to face it. Support groups can also be helpful.",
    "sadness": "Engage in activities that you enjoy and connect with loved ones for emotional support.",
    "lonely": "Join social groups or online communities that align with your interests.",
    "isolation": "Consider reaching out to friends or family members and engaging in community activities.",
    "overwhelmed": "Break tasks into smaller steps, delegate responsibilities, and take breaks when needed.",
    "worthless": "Remember that self-worth is not defined by external factors. Engage in activities that boost self-esteem.",
    "self-harm": "Reach out to a trusted friend or counselor. There are healthier ways to cope with difficult emotions.",
    "insomnia": "Maintain a consistent sleep schedule and avoid screens before bedtime. Consider relaxation techniques.",
    "fatigue": "Ensure you're getting enough rest and balanced nutrition. If persistent, consult a healthcare professional.",
    "anger": "Practice anger management techniques like deep breathing and counting to ten before reacting.",
    "guilt": "Acknowledge the source of your guilt, make amends if possible, and focus on self-forgiveness."
}

# Function: Clean Text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

# Function: Analyze Mental Health
def analyze_mental_health(text):
    cleaned_text = clean_text(text)
    sentiment_score = sia.polarity_scores(cleaned_text)
    detected_keywords = [word for word in MENTAL_HEALTH_KEYWORDS if word in cleaned_text]

    mental_health_status = (
        "Potential Mental Health Concern"
        if sentiment_score['compound'] <= -0.3 or len(detected_keywords) > 2
        else "No Significant Concern"
    )

    solutions = [MENTAL_HEALTH_SOLUTIONS[keyword] for keyword in detected_keywords]
    return {
        "Sentiment Score": sentiment_score,
        "Detected Keywords": detected_keywords,
        "Mental Health Status": mental_health_status,
        "Solutions": solutions or ["Keep monitoring your mental health and focus on positive activities."]
    }

# Function: Extract Audio Features
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        return np.concatenate((mfcc, chroma, spectral_contrast))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function: Predict Speech Tone
def predict_speech_tone(audio_path):
    feature = extract_features(audio_path)
    if feature is not None:
        prediction = model.predict([feature])
        return "Happy" if prediction[0] == 1 else "Neutral" if prediction[0] == 0 else "Sad"
    return "Error in processing audio file."

# Function: Psychological Evaluation
def psychological_evaluation():
    print("Welcome to the Psychological Evaluation")
    questions = [
        "On a scale of 1 to 5, how often do you feel stressed in a day?",
        "On a scale of 1 to 5, how well do you sleep at night?",
        "On a scale of 1 to 5, how often do you feel motivated to accomplish your daily goals?",
        "On a scale of 1 to 5, how often do you feel overwhelmed by responsibilities?",
        "On a scale of 1 to 5, how often do you feel content with your life?"
    ]
    responses = {}
    for i, question in enumerate(questions, start=1):
        while True:
            try:
                response = int(input(f"{question} (1-5): "))
                if 1 <= response <= 5:
                    responses[f"Question {i}"] = response
                    break
                else:
                    print("Answer must be between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 5.")

    print("\nAnalysis:")
    print("Stress Level:", "High" if responses["Question 1"] > 3 else "Low")

# Emotion Detection with Webcam
'''
This particualr fuction is not working currently and as i have not taken proper dataset or not train it properly
'''
def emotion_detection():
    model = load_model("emotion_detection_model.h5")
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            face = cv2.resize(roi_gray, (48, 48))
            face = face.astype("float32") / 255
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)
            prediction = model.predict(face)
            emotion = emotion_labels[np.argmax(prediction)]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Welcome to the Mental Health Analysis App")
    print("Please choose an input method:")
    print("1. Enter your thoughts via text")
    print("2. Use the camera for emotion detection")
    
    while True:
        try:
            choice = int(input("Enter your choice (1 or 2): "))
            if choice == 1:
                text_input = input("Tell me about your problems: ")
                print("\nText Analysis Results:")
                print(analyze_mental_health(text_input))
                break
            elif choice == 2:
                print("\nStarting emotion detection using your webcam...")
                emotion_detection()
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter 1 or 2.")

    print("\nProceeding to Psychological Evaluation...\n")
    psychological_evaluation()
