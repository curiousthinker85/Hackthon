import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st

# Load the pre-trained emotion detection model
emotion_model = load_model('C:/Users/Hp/OneDrive/Documents/Productivity tool/Hackthon/fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the avatars for different emotions
avatars = {}
for label in emotion_labels:
    avatars[label] = Image.open(f"avatars/{label.lower()}.png")

# Function to detect facial expressions
def detect_emotion(face_img):
    # Preprocess the image for the emotion detection model
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img.astype("float") / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)

    # Predict the emotion using the emotion detection model
    preds = emotion_model.predict(face_img)[0]
    emotion_label = emotion_labels[np.argmax(preds)]
    return emotion_label

# Function to create a simple avatar based on the detected emotion
def create_avatar(emotion_label):
    avatar = avatars.get(emotion_label)

    if avatar is None:
        st.write(f"Avatar for {emotion_label} not found.")
        return None

    return np.array(avatar)

def main():
    st.title("Emotion Detection and Avatar")

    option = st.sidebar.radio("Choose an option:", ("Live Detection", "Image Detection"))

    if option == "Live Detection":
        st.write("Click the 'Start' button to begin live emotion detection.")
        start_button = st.sidebar.button("Start")
        stop_button = st.sidebar.button("Stop")

        if start_button:
            cap = cv2.VideoCapture(0)

            while not stop_button:
                ret, frame = cap.read()

                # Detect faces in the frame
                faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    emotion_label = detect_emotion(face_img)

                    # Draw the face rectangle and emotion label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Show the avatar on the side
                avatar = create_avatar(emotion_label)
                if avatar is not None:
                    avatar = cv2.resize(avatar, (100, 100))
                    frame[10:110, 10:110] = cv2.cvtColor(avatar, cv2.COLOR_RGB2BGR)

                # Display the frame with live emotion detection
                cv2.imshow('Emotion Detection and Avatar', frame)

                if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                    break

            cap.release()
            cv2.destroyAllWindows()

    elif option == "Image Detection":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image)
            faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_img = image[y:y+h, x:x+w]
                emotion_label = detect_emotion(face_img)

                # Draw the face rectangle and emotion label
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(image, emotion_label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Show the avatar below the image
                avatar = create_avatar(emotion_label)
                if avatar is not None:
                    st.image(avatar, caption=f"Emotion: {emotion_label}", width=100)

            # Display the image with emotion detection results
            st.image(image, caption="Emotion Detection Result", use_column_width=True)

if __name__ == "__main__":
    main()
