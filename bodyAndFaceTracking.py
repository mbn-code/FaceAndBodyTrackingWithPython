try:
    import cv2
    import os


    # here are all the haar cascades
    # https://github.com/opencv/opencv/tree/master/data/haarcascades

    # I --- body haar cascades .xml files

    cascPath_pro = os.path.dirname(cv2.__file__) + "upperbody.xml"
    upper_cas = cv2.CascadeClassifier("upperbody.xml")

    cascPath = os.path.dirname(cv2.__file__) + "face_default.xml"
    cascPath_pro = os.path.dirname(cv2.__file__) + "faceProfile_extended.xml"
    faceCascade = cv2.CascadeClassifier("face_default.xml")
    faceCascadeProfile = cv2.CascadeClassifier("faceProfile_extended.xml")

    cascPath_pro = os.path.dirname(cv2.__file__) + "frontalFaceCloser.xml"
    frontalFaceCloser = cv2.CascadeClassifier("frontalFaceCloser.xml")

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

        # I ----- draw rectangles around body --- I

        for (x,y,h,w) in upperbody:
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0,0,255), 2)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,255), 2)
            # Display the resulting frame

        for (x,y,h,w) in faces_profile:
            cv2.rectangle(frame, (x, y), (x + w, y + h),(255,255,0), 1)

        
        for (x,y,h,w) in frontalFaceCloser_detect:
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0,200,0), 1)


        # I -------- Check for face detected ---------- I
        
        if len(upperbody) > 0:    
            print("Body detected " + str(upperbody))
        else:
            body_detect = ""

        if len(faces_profile) > 0:
            print("face(s) profile detected " + str(faces_profile))
        else:
            body_detect = ""

        if len(faces) > 0:    
            print("Face(s) frontal detected " + str(faces))
        else:
            body_detect = ""

        # 
        if body_detect:
            print("Body upper detected " + str(body_detect))
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
except:
    if KeyboardInterrupt:
        print("Deactivating lock feature")