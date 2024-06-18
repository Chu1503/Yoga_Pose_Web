import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp

def main():
    st.title("Pose Detection with Streamlit")

    # Define MediaPipe pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Define WebRTC video streamer
    webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=pose_detection)

    if webrtc_ctx.video_processor:
        st.write("WebRTC connected!")

def pose_detection():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    def processor(frame):
        # Resize the frame to 1280x720
        frame = cv2.resize(frame, (1280, 720))

        # Convert the frame to RGB (required by MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get the pose landmarks
        results = pose.process(frame_rgb)

        # Draw lines on the frame
        if results.pose_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return frame

    return processor

if __name__ == "__main__":
    main()