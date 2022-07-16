import cv2
import os

# here are all the haar cascades
# https://github.com/opencv/opencv/tree/master/data/haarcascades

# I --- Font and side haar cascades .xml files
cascPath = os.path.dirname(cv2.__file__) + "face_default.xml"
cascPath_pro = os.path.dirname(cv2.__file__) + "faceProfile_extended.xml"
faceCascade = cv2.CascadeClassifier("face_default.xml")
faceCascadeProfile = cv2.CascadeClassifier("faceProfile_extended.xml")

cascPath_pro = os.path.dirname(cv2.__file__) + "frontalFaceCloser.xml"
frontalFaceCloser = cv2.CascadeClassifier("frontalFaceCloser.xml")

# I -- Video capture device to use 
video_capture = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # I ----- Detect faces using the haar cascades .xml files

    faces = faceCascadeProfile.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=10,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    faces_profile = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=11,
                                         minSize=(30, 35),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    frontalFaceCloser_detect = frontalFaceCloser.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=2,
                                         minSize=(40, 45),
                                         )

    # I ----- draw rectangles around front and profile face --- I

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,255), 2)
        # Display the resulting frame
    
    for (x,y,h,w) in faces_profile:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(255,255,0), 1)

    
    for (x,y,h,w) in frontalFaceCloser_detect:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,200,0), 1)

    # check if system is mac or linux or windows 
    # chcek if win 10 or not

    if os.name == 'Darwin':
        if len(faces) or len(faces_profile) > 0:
            os.system("""osascript -e 'tell application "Terminal" to display alert "Face Detected" as warning'""")
    elif os.name == 'nt':
        print("Operating System is Windows")



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