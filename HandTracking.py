import cv2
import time

handCascade = cv2.CascadeClassifier("handTracking.xml")

palmCascade = cv2.CascadeClassifier("handPalm.xml")

frontalFace = cv2.CascadeClassifier("frontalFaceCloser.xml")

video_capture = cv2.VideoCapture(2)

while 1:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    hands = handCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 60),
    )


    handPalm = palmCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 60),
    )

    frontalFaceCloser_detect = frontalFace.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=3,
                                         minSize=(40, 45),
                                         )

    for (x,y,w,h) in frontalFaceCloser_detect:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (225,50,20), 3)

    if len(frontalFaceCloser_detect) > 0:
        for (x, y, w, h) in handPalm:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (180, 0, 20), 3)

        for (x, y, w, h) in hands:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (20, 0, 180), 3)

# These are the functions for unlocking the hand and activating the hands with the face
        
    


    # Display the resulting frame
    width = 1280
    height = 750
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
    cv2.imshow("Face Hand Unlock", resized)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()