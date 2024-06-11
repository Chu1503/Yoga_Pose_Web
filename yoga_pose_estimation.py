import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def pose_detector(pose_name):
    prev_time = 0
    fps = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Calculate angle
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                # Calculate accuracy based on the pose
                if pose_name == "Warrior Pose":
                    left_elbow_accuracy = 100 - abs(160 - left_elbow_angle)
                    right_elbow_accuracy = 100 - abs(160 - right_elbow_angle)
                    left_knee_accuracy = 100 - abs(170 - left_knee_angle)
                    right_knee_accuracy = 100 - abs(170 - right_knee_angle)  # Different accuracy calculation for Warrior Pose
                    overall_accuracy = (right_elbow_accuracy + left_elbow_accuracy + left_knee_accuracy + right_knee_accuracy) / 4
                    elbow_range = (160, 180)  # Angle range for rendering
                elif pose_name == "Plank Pose":
                    left_elbow_accuracy = 100 - abs(170 - left_elbow_angle)
                    right_elbow_accuracy = 100 - abs(170 - right_elbow_angle)
                    left_knee_accuracy = 100 - abs(180 - left_knee_angle)
                    right_knee_accuracy = 100 - abs(180 - right_knee_angle)  # Different accuracy calculation for Plank Pose
                    overall_accuracy = (right_elbow_accuracy + left_elbow_accuracy + left_knee_accuracy + right_knee_accuracy) / 4
                    elbow_range = (170, 180)  # Angle range for rendering

                # Visualize angle
                font_scale = 1.0
                
                cv2.rectangle(image, (0, 0), (300, 200), (0, 0, 0), -1)

                cv2.putText(image, f'{left_elbow_angle:.2f}', 
                            tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{right_elbow_angle:.2f}', 
                            tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{left_knee_angle:.2f}', 
                            tuple(np.multiply(left_knee, [1280, 720]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{right_knee_angle:.2f}', 
                            tuple(np.multiply(right_knee, [1280, 720]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,255,255), 2, cv2.LINE_AA)

                # Display accuracies
                cv2.putText(image, f'Overall Accuracy: {overall_accuracy:.2f}%', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Left Elbow Accuracy: {left_elbow_accuracy:.2f}%', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Right Elbow Accuracy: {right_elbow_accuracy:.2f}%', (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Left Knee Accuracy: {left_knee_accuracy:.2f}%', (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Right Knee Accuracy: {right_knee_accuracy:.2f}%', (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                pass

            # Render detections based on pose and angle range
            if elbow_range[0] < left_elbow_angle < elbow_range[1]:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                    )
            else:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2) 
                                    )

            # Calculate and display FPS
            current_time = time.time()
            elapsed_time = current_time - prev_time
            fps = 1 / elapsed_time
            prev_time = current_time

            cv2.putText(image, f'FPS: {int(fps)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield frame

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    st.title("Yoga Pose Estimation")
    st.write("Choose a pose and see the real-time estimation through your webcam.")

    pose_name = st.selectbox("Select a pose:", ["Warrior Pose", "Plank Pose"])

    webcam_in_use = False  # Variable to track webcam usage

    if st.button("Start Pose Estimation"):
        webcam_in_use = True  # Set webcam usage to true when button is clicked

    frame_placeholder = st.empty()

    if webcam_in_use:
        close_button = st.button("Close Webcam")  # Display close button when webcam is in use
        for frame in pose_detector(pose_name):
            frame_placeholder.image(frame, channels="BGR")
            if close_button:  # Check if close button is clicked
                webcam_in_use = False
                break

if __name__ == "__main__":
    main()