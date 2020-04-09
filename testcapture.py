import cv2
cap = cv2.VideoCapture(1)
while(True):
    ret, frame = cap.read()
   
    print(ret)
    cv2.imshow('frame1', frame)
