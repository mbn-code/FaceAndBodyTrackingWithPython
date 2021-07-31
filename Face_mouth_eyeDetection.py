import cv2

faceCascade_Default = cv2.CascadeClassifier("face_default.xml")

faceCascade_profile = cv2.CascadeClassifier("face_side.xml")

eyeCascade = cv2.CascadeClassifier("eyeDetection.xml")

mouthDetection = cv2.CascadeClassifier("mouthDetection.xml")

video_capture = cv2.VideoCapture(0)

face_detect = "Face detected"

while 1:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_1 = faceCascade_Default.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=40,
        minSize=(30, 35),
    )

    facesProfile = faceCascade_profile.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=40,
        minSize=(30, 35),
    )

    eyeCascade_eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=40,
        minSize=(2, 3),
    )

    mouthDetection_mouth = mouthDetection.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=40,
        minSize=(2, 3),
    )



    # Draw a rectangle around the faces
    for (x, y, w, h) in faces_1:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 0, 20), 2)

    for (x,y,w,h) in facesProfile:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (20, 0, 180), 2)

    for (x,y,w,h) in eyeCascade_eyes:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 200, 0), 1)

    for (x,y,w,h) in mouthDetection_mouth:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 200, 200), 1)


    if len(faces_1) > 0:
        face_detect = "Face detected"
    else:
        face_detect = "no face detected"
    print(face_detect)

    # Display the resulting frame
    cv2.imshow('GUI IMG READ', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
