import cv2
numberPlateCascade=cv2.CascadeClassifier("C:/Users/AMAN/Desktop/haarcascade_russian_plate_number.xml")
path='C:/Users/AMAN/Desktop/car.jpg'
minArea=500
color=(255,0,255)
img=cv2.imread(path)
frameWidth=360
frameHeight=360
while True:
    # success,img=cap.read()
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    numberPlates=numberPlateCascade.detectMultiScale(imgGray,1.1,4)
    for (x,y,w,h) in numberPlates:
        area=w*h
        if area>minArea:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            cv2.putText(img,"Number Plate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            imgRoi=img[y:y+h,x:x+w]
            cv2.imshow("ROI",imgRoi)
    cv2.imshow("video",img)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break