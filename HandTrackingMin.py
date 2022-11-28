# import cv2
# import mediapipe
#

#
# cap = cv2.VideoCapture(0)
#
# while True:
#     success,img = cap.read()
#
#     cv2.imshow("Image",img)
#     cv2.waitKey(0)

import urllib.request
import cv2
import mediapipe as mp
import time
import numpy as np
import imutils

url='http://192.168.1.105:8080/shot.jpg'

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    img = imutils.resize(img, width=750)

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                print(id, cx,cy)
                if id ==21:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

            mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("CameraFeed",img)
    if ord('q') ==  cv2.waitKey(1):
        exit(0)