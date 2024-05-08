import cv2
import mediapipe as mp
import numpy as np
import os

# Set up MediaPipe Pose

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1)

mpDraw = mp.solutions.drawing_utils

# Paths to existing data files
test_name = input("Enter the test directory name to load: ")
test_dir = os.path.join(os.getcwd(), test_name)
avi_path = os.path.join(test_dir, "color_video.avi")
npz_path = os.path.join(test_dir, "depth_data.npz")

# Load the saved depth data
depth_data = np.load(npz_path)
depth_frames = depth_data['depth_frames']
timestamps = depth_data['timestamps']

# Open the color video file
cap = cv2.VideoCapture(avi_path)
depth_frame_index = 0

# Font settings for display
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (0, 255, 0)
thickness = 1

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or depth_frame_index >= len(depth_frames):
            break

        # Get the corresponding depth frame
        depth_frame = depth_frames[depth_frame_index]

        # Process the color frame with MediaPipe Pose
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Draw pose landmarks
        if results.pose_landmarks:
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display timestamp
        timestamp = timestamps[depth_frame_index]
        cv2.putText(frame, f"Timestamp: {timestamp}", (20, 50), font, fontScale, color, thickness)

        # Show the frame with the overlaid pose landmarks
        cv2.imshow('Replay with Pose Landmarks', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Move to the next depth frame
        depth_frame_index += 1

finally:
    cap.release()
    cv2.destroyAllWindows()
