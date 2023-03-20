import cv2
import numpy as np
import threading

# Define the face detection function
def detect_faces(frame, net, min_confidence):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

# Define the thread function
def process_frame(frame, net, min_confidence):
    # Perform face detection on the frame
    detect_faces(frame, net, min_confidence)

# Load the face detection model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Set the minimum confidence threshold for face detection
min_confidence = 0.5

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Loop over the frames from the video stream
while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()

    if not ret:
        break

    # Create a thread for processing the current frame
    thread = threading.Thread(target=process_frame, args=(frame, net, min_confidence))

    # Start the thread
    thread.start()

    # Wait for the thread to finish beforse processing the next frame
    thread.join()

    # Display the frame
    cv2.imshow('frame', frame)

    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
