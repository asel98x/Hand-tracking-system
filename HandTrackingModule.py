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

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition (self, img, handNo=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList











def main():
    pTime = 0
    cTime = 0
    detector = handDetector()

    while True:
        imgPath = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        img = imutils.resize(img, width=750)

        img = detector.findHands(img,draw=False)
        lmList = detector.findPosition(img,draw=False)

        if len(lmList)!=0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("CameraFeed", img)
        if ord('q') == cv2.waitKey(1):
            exit(0)

if __name__=="__main__":
    main()
