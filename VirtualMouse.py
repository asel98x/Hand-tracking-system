import cv2
import numpy as np
import HandTrackingModule2 as htm
import time
import autopy
import urllib.request
import imutils

url = 'http://192.168.1.100:8080/shot.jpg'
pTime = 0
cTime = 0
detector = htm.handDetector(maxHands=1)
wCam = 640
hCam = 480
frameR=100
smoothening=7
pLocx,pLocy = 0,0
cLocx,cLocy = 0,0


while True:
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    img = imutils.resize(img, wCam,hCam)
    wScr,hScr = autopy.screen.size()
    # print(wScr,hScr)

    img = detector.findHands(img)
    lmList,bbox = detector.findPosition(img)
    #print(lmList)
    if len(lmList) != 0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        #print(x1,y1,x2,y2)

        fingers = detector.fingersUp()
        #print(fingers)


        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),(255, 0, 255), 2)

        if fingers[1]==1 and fingers[2]==0:

            x3 = np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            cLocx = pLocx + (x3-pLocx)/smoothening
            cLocy = pLocy + (y3 - pLocy)/smoothening

            autopy.mouse.move(wScr-cLocx,cLocy)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            pLocx,pLocy = cLocx,cLocy

        if fingers[1] == 1 and fingers[2] == 1:
            length,img,lineInfo = detector.findDistance(8,12,img)
            # print(length)

            if length<40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (255, 0, 5), cv2.FILLED)
                autopy.mouse.click()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


    cv2.imshow("CameraFeed", img)
    if ord('q') == cv2.waitKey(1):
        exit(0)
