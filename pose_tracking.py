import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import datetime as dt
import csv
import os

# Ask the user for a test name
test_name = input("Enter the name of the test: ")

# Create a new directory for this test if it doesn't exist
test_dir = os.path.join(os.getcwd(), test_name)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# File paths for saving data
csv_file_path_deprojected = os.path.join(test_dir, "pose_tracking_data_deprojected.csv")
csv_file_path_pixel = os.path.join(test_dir, "pose_tracking_data_pixel.csv")
color_video_path = os.path.join(test_dir, "color_video.avi")
depth_data_path = os.path.join(test_dir, "depth_data.npz")

# Configure depth and color streams from the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming from RealSense
pipeline.start(config)

# Set up MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mpDraw = mp.solutions.drawing_utils

# Get depth scale and intrinsics
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Font settings for display
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (0, 255, 0)
thickness = 1

# List to keep track of which landmark IDs are present
landmark_ids = [f"ID_{i}" for i in range(33)]

# Initialize lists to store data
data_deprojected = []
data_pixel = []
depth_frames = []
frame_timestamps = []

# Initialize video writer for color frames
fourcc = cv2.VideoWriter_fourcc(*'XVID')
color_video_out = cv2.VideoWriter(color_video_path, fourcc, 30, (640, 480))

try:
    while True:
        start_time = dt.datetime.now()

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Save raw color video frames
        color_video_out.write(color_image)

        # Save raw depth frame and timestamp
        depth_frames.append(depth_image)
        frame_timestamps.append(start_time.strftime("%Y-%m-%d %H:%M:%S"))

        # Process the color frame with MediaPipe Pose
        results = pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        frame_data_deprojected = {"Frame": start_time.strftime("%Y-%m-%d %H:%M:%S")}
        frame_data_pixel = {"Frame": start_time.strftime("%Y-%m-%d %H:%M:%S")}
        for landmark_id in landmark_ids:
            frame_data_deprojected[landmark_id] = "NaN"
            frame_data_pixel[landmark_id] = "NaN"

        # Draw pose landmarks
        if results.pose_landmarks:
            mpDraw.draw_landmarks(color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            for id, lm in enumerate(results.pose_landmarks.landmark):
                cx, cy = int(lm.x * color_image.shape[1]), int(lm.y * color_image.shape[0])

                # Ensure coordinates are within the bounds of the depth image
                if 0 <= cx < color_image.shape[1] and 0 <= cy < color_image.shape[0]:
                    depth = depth_image[cy, cx] * depth_scale
                    
                    # Deproject pixel to point in 3D (meters)
                    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], depth)

                    # Update the frame data with both deprojected and pixel coordinates
                    frame_data_deprojected[f"ID_{id}"] = f"{point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f}"
                    frame_data_pixel[f"ID_{id}"] = f"{cx}, {cy}, {depth:.9f}"

                    # Display XYZ coordinates with ID
                    cv2.putText(color_image, f"ID {id}: ({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f})", (cx, cy), font, fontScale, color, thickness)
                else:
                    cv2.putText(color_image, f"ID {id}: Out of bounds", (cx, cy), font, fontScale, color, thickness)

        # Save frame data to the data lists
        data_deprojected.append(frame_data_deprojected)
        data_pixel.append(frame_data_pixel)

        # Calculate FPS
        fps = int(1 / (dt.datetime.now() - start_time).total_seconds())
        cv2.putText(color_image, f"FPS: {fps}", (20, 80), font, fontScale, color, thickness)

        # Show the frame
        cv2.imshow('RealSense Pose Tracking', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Write deprojected data to CSV
    with open(csv_file_path_deprojected, "w", newline='') as csvfile:
        fieldnames = ["Frame"] + landmark_ids
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_deprojected)

    # Write pixel data to a separate CSV
    with open(csv_file_path_pixel, "w", newline='') as csvfile:
        fieldnames = ["Frame"] + landmark_ids
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_pixel)

    # Save depth frames and timestamps to a .npz file
    np.savez(depth_data_path, depth_frames=depth_frames, timestamps=frame_timestamps)

    # Release video writer and stop streaming
    color_video_out.release()
    pipeline.stop()
    cv2.destroyAllWindows()

