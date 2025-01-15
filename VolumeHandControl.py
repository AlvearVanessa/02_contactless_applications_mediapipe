"""
30 01 2023 12:33h CET
Contactless interaction human-computer to control the volume of the computer
in real-time based on the distance of the tip index and tip thumb

Source:
https://www.youtube.com/watch?v=9iEPzbG-xLE
"""

import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

####################################
#Parameters for camera size
wCam, hCam = 640, 480
####################################

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# Setting sizes adn 3 and 4 are Id's
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

# Initialization
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        #print(lmList[4], lmList[8])


        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2


        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

        #Line between these points
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        #Creating a circle in the center of the line
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        #print(length)

        # Hand range 50 - 300
        #Volume Range -28 - 0
        vol = np.interp(length, [15, 230], [minVol, maxVol])  # length is the value we want to convert
        volBar = np.interp(length, [15, 230], [400, 150])
        volPer = np.interp(length, [15, 230], [0, 100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)


    cv2.rectangle(img, (50, 150), (85,400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    # Adding frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
