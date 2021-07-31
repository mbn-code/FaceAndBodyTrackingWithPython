import cv2

faceCascade_Default = cv2.CascadeClassifier("face_default.xml")

faceCascade_profile = cv2.CascadeClassifier("face_side.xml")

eyeCascade = cv2.CascadeClassifier("eyeDetection.xml")

mouthDetection = cv2.CascadeClassifier("mouthDetection.xml")

video_capture = cv2.VideoCapture(0)

face_detect = "Face detected"

facesProfile_detect = "Face detected"

while 1:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_1 = faceCascade_Default.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=25,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    facesProfile = faceCascade_profile.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=55,
        minSize=(30, 35),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces_1:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 0, 20), 1)

    for (x,y,w,h) in facesProfile:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (20, 0, 180), 1)

    if len(faces_1) > 0:
        face_detect = "Face (front) detected"
    else:
        face_detect = "no face (front) detected"
    print(face_detect)

    if len(facesProfile) > 0:
        facesProfile_detect = "Face (side) detected"
    else:
        facesProfile_detect = "Not face (side) detected"
    print(facesProfile_detect)


    # Display the resulting frame
    cv2.imshow('GUI IMG READ', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
