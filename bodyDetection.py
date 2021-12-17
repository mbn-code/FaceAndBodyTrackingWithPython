import cv2
import os
import numpy as np


# here are all the haar cascades
# https://github.com/opencv/opencv/tree/master/data/haarcascades

# I --- body haar cascades .xml files
cascPath_pro = os.path.dirname(cv2.__file__) + "fullbody.xml"
Fullbody_cas = cv2.CascadeClassifier("fullbody.xml")

cascPath_pro = os.path.dirname(cv2.__file__) + "upperbody.xml"
upper_cas = cv2.CascadeClassifier("upperbody.xml")

cascPath_pro = os.path.dirname(cv2.__file__) + "lowerbody.xml"
lower_cas = cv2.CascadeClassifier("lowerbody.xml")


# I -- Video capture device to use 
video_capture = cv2.VideoCapture(0)

# - veriable for if the face is detected
body_detect = "body"


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # I ----- Detect faces using the haar cascades .xml files


    upperbody = upper_cas.detectMultiScale(
                                        gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(60, 60),
                                        flags=cv2.CASCADE_SCALE_IMAGE)


    # I ----- draw rectangles around body --- I

    for (x,y,h,w) in upperbody:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,0,255), 2)
        print(upperbody)

    # I -------- Check for face detected ---------- I

    import random
    
    if len(upperbody) > 0:    
        print("body detected " + " -- ID: " + str(random.randint(1000,9999)))        
    else:
        body_detect = ""

    # 
    if body_detect:
        print(body_detect)
    else:
        pass

    # -------------- Show the image ----------------
    width = 900
    height = 750
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
    cv2.imshow("Body tracking", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()