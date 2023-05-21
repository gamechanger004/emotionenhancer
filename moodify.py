import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import pywhatkit
# Load the emotion classification model and labels
model  = load_model("demo.h5")
label = np.load("labels.npy")
# Initialize the holistic and hand detection models from MediaPipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils
# Set up the Streamlit app
st.header("Moodify-Emotion Enhancer")
# Check if the emotion has already been captured
if "run" not in st.session_state:
    st.session_state["run"] = "true"
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion=""
# If the emotion has not been captured or saved, set the app to capture it
if not(emotion):
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"
# Define a class for processing the video frames
class EmotionProcessor:
    def recv(self,frame):
        frm=frame.to_ndarray(format="bgr24")
        ##############################
        # Flip the frame horizontally to mirror the user's movements
        frm=cv2.flip(frm,1)
        # Use the holistic model to detect landmarks on the user's face and hands
        res=holis.process(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB))
        lst = []
        # If landmarks are detected on the face, add the x and y distances between each landmark and the second landmark to a list
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)
            # If landmarks are detected on the left hand
            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)
            # If landmarks are detected on the right hand
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)
            # Reshape the list and use the model to predict the user's emotion from the landmarks
            lst = np.array(lst).reshape(1,-1)
            pred = label[np.argmax(model.predict(lst))]
            print(pred)
            # Draw the predicted emotion on the video frame
            cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)
            np.save("emotion.npy", np.array([pred]))
        # Draw the face, left hand, and right hand landmarks on the video frame
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1,circle_radius=1),connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
        ###################################
        # Return the processed video frame as a VideoFrame object
        return av.VideoFrame.from_ndarray(frm, format="bgr24")
if  st.session_state["run"] != "false":
    webrtc_streamer(key="key",desired_playing_state= True,video_processor_factory=EmotionProcessor)
# Create a Streamlit button to capture the user's emotion
btn = st.button("caputure my emotion")
if btn:
    # Check if an emotion has already been captured
    if not(emotion):
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        # If an emotion has been captured, send a WhatsApp message with the emotion result
        phone_number = "<+918919407781>"

        if emotion == "happy":
            message = "He is feeling happy share more happiness."
        elif emotion == "sad":
            message = "He is feeling sad make him feel better."
        elif emotion == "angry":
            message = "He is feeling angry make him cool down."
        elif emotion == "neutral":
            message = "He is okay dont worry."
        elif emotion == "rockrock":
            message = "he is super active"
        else:
            message = "He is feeling " + emotion + "."
        # Send the WhatsApp message
        pywhatkit.sendwhatmsg_instantly(phone_number, message)
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"