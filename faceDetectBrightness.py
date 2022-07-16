import cv2
import os
import screen_brightness_control as sbc
# here are all the haar cascades
# https://github.com/opencv/opencv/tree/master/data/haarcascades


cascPath_pro = os.path.dirname(cv2.__file__) + "frontalFaceCloser.xml"
frontalFaceCloser = cv2.CascadeClassifier("frontalFaceCloser.xml")

# I -- Video capture device to use 
video_capture = cv2.VideoCapture(2)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # I ----- Detect faces using the haar cascades .xml files


    frontalFaceCloser_detect = frontalFaceCloser.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=2,
                                         minSize=(40, 45),
                                         )

    
    for (x,y,h,w) in frontalFaceCloser_detect:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,200,0), 1)

    if frontalFaceCloser:
        for monitor in sbc.list_monitors():
            print(monitor, ':', sbc.get_brightness(display=monitor), '%')

    # -------------- Show the image ----------------
    width = 1280
    height = 750
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
    cv2.imshow("Face tracking", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


video_capture.release()
cv2.destroyAllWindows()