import cv2
import random


handCascade = cv2.CascadeClassifier("handTracking.xml")

palmCascade = cv2.CascadeClassifier("handPalm.xml")

frontalFaceCloser = cv2.CascadeClassifier("frontalFaceCloser.xml")


video_capture = cv2.VideoCapture(0)

while 1:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    hands = handCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 60),
    )


    handPalm = palmCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 60),
    )

    frontalFaceCloser_detect = frontalFaceCloser.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(40, 45),
                                         )

# These are the functions for unlocking the hand and activating the hands with the face

    def unlock_hand():
        for i in range(10): 
            for (x, y, w, h) in handPalm:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (180, 0, 20), 3)
                print("Unlocked -- " + str(random.randint(1000,9999)))

    def hands_activation():
        for (x, y, w, h) in hands:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (20, 0, 180), 3)
        
        if len(hands) > 0:
            unlock_hand()

    for (x,y,h,w) in frontalFaceCloser_detect:
        x = cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 3)

    if len(frontalFaceCloser_detect) > 0:
        hands_activation()
    else:
        pass
        
    
    # Display the resulting frame
    width = 1110
    height = 900
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
    cv2.imshow("Face Hand Unlock", resized)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()