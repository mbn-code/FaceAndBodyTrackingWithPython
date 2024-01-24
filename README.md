# Face Tracking System with Haar Cascades
Note: This readme is written around the Main_main python file. 
This Python script utilizes the OpenCV library to implement a simple face tracking system using Haar cascades. Haar cascades are trained classifiers that can be used for object detection. In this case, we are using Haar cascades to detect faces in a video stream. The system draws rectangles around detected faces and provides a basic function when a face is detected.

## Setup

Before running the script, make sure to install the required libraries. You can install OpenCV using the following:

```bash
pip install opencv-python
```

## Haar Cascades

The script uses Haar cascade files for detecting faces. The cascade files can be found in the OpenCV GitHub repository [here](https://github.com/opencv/opencv/tree/master/data/haarcascades). The following Haar cascade files are used in this script:

- `face_default.xml`: Cascade file for detecting frontal faces.
- `faceProfile_extended.xml`: Cascade file for detecting profile faces.
- `frontalFaceCloser.xml`: Additional cascade file for detecting closer frontal faces.

## Usage

1. Ensure that your webcam is connected and functional.
2. Run the script.

The script will continuously capture video frames from the specified video capture device (in this case, device index 2). Detected faces will be outlined with rectangles of different colors, and a simple function `FaceDetected` will be called when a face is detected, printing a message with a random ID.

## Configuration

You can adjust the script's parameters for face detection by modifying the following:

- `scaleFactor`: Parameter specifying how much the image size is reduced at each image scale.
- `minNeighbors`: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
- `minSize`: Minimum possible object size. Objects below this size will not be detected.

## Dependencies

- Python 3.x
- OpenCV

## Notes

- Press 'q' to exit the video stream.
