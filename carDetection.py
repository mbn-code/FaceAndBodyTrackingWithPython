import cv2

faceCascade_Default = cv2.CascadeClassifier("face_default.xml")

faceCascade_profile = cv2.CascadeClassifier("carCascade.xml")

numberplate_car = cv2.CascadeClassifier("numberplate_detection.xml")

video_capture = cv2.VideoCapture(0)

while 1:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade_Default.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=21,
        minSize=(30, 35),
    )

    facesProfile = faceCascade_profile.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=12,
        minSize=(30, 35),
    )

    numberplate_car = faceCascade_profile.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=20,
        minSize=(30, 35),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 0, 20), 2)

    for (x,y,w,h) in facesProfile:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (20, 0, 180), 2)


    for (x,y,w,h) in numberplate_car:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (20, 200, 180), 2)

    # Display the resulting frame
    cv2.imshow('GUI IMG READ', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
