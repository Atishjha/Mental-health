import cv2
import numpy as np

# Load pre-trained emotion detection model (simple CNN based)
# Download the model from a source like fer2013 or custom Keras-based CNN
# This is an example of loading a .h5 model - modify based on your choice

from keras.models import load_model

# Load pre-trained model (FER or a CNN model)
model = load_model("emotion_detection_model.h5")

# Dictionary to label all emotion expressions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start webcam feed
cap = cv2.VideoCapture(0)

# Preprocess input function (match your model's input shape)
def preprocess_input(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))  # Resize to 48x48 (FER)
    face = face.astype("float32") / 255  # Normalize pixels
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)  # Shape (1, 48, 48, 1) for CNN
    return face

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and detect faces using Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        preprocessed_face = preprocess_input(roi_gray)

        # Predict emotion using model
        prediction = model.predict(preprocessed_face)
        max_index = int(np.argmax(prediction))
        emotion = emotion_labels[max_index]

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display output in window
    cv2.imshow('Facial Expression Recognition', frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
