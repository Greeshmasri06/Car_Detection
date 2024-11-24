import cv2
import time
import sys  # Import sys for exit() function
import os  # For file path checking

# File paths for car classifier and video
car_classifier_path = r'C:\Users\lenovo\Desktop\NIT FILES\21st- python to mysql connection\haar cascade classifier basic project\Haarcascades\haarcascade_car.xml'
video_path = r'C:\Users\lenovo\Desktop\NIT FILES\21st- python to mysql connection\haar cascade classifier basic project\Haarcascades\video.mp4'

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}. Make sure the path is correct.")
    sys.exit()  # Exit if video path is incorrect
else:
    print(f"Video file found at: {video_path}")

# Load the car cascade classifier
car_classifier = cv2.CascadeClassifier(car_classifier_path)

# Check if the classifier is loaded correctly
if car_classifier.empty():
    print(f"Error: Could not load the car classifier at {car_classifier_path}. Make sure the path is correct.")
    sys.exit()  # Using sys.exit() to exit the program

# Load the video
cap = cv2.VideoCapture(video_path)

# Check if the video file is loaded correctly
if not cap.isOpened():
    print(f"Error: Could not open the video at {video_path}. Make sure the file path is correct and video format is supported.")
    sys.exit()  # Using sys.exit() to exit the program

print("Video opened successfully. Starting car detection...")

# Main loop to read and process video frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame or video has ended.")
        break

    # Convert the frame to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the grayscale image
    cars = car_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Draw rectangles around the detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Display the frame with the detected cars
    cv2.imshow('Cars Detection', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 113 is the ASCII code for 'q'
        print("Exiting...")
        break

    # Optional: Add a small delay between frames (to make the processing smoother)
    time.sleep(0.05)  # You can adjust or remove this if it's slowing down the process

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()














