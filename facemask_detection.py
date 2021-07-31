import cv2
import sys

facemaskCascade = cv2.CascadeClassifier("facemask_frontal.xml")

video_capture = cv2.VideoCapture(0)

while 1:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facemaskCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120),

    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 0, 20), 2)

    # Display the resulting frame
    cv2.imshow('Face mask detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
