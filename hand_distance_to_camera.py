"""
Thur, 21 03 2024 13:51h CET
Hand landmarks Module
Source: https://www.computervision.zone/
"""
import cv2
import HandTrackingModule_black_camera_hand_distance as htm
from HandTrackingModule_black3 import handDetector
import math
import numpy as np



# Webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = htm.handDetector(detectionCon=0.8, maxHands=1)


# Find Function
# x is the raw distance y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C



def putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),
                colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,
                offset=10, border=None, colorB=(0, 255, 0)):
    """
    Creates Text with Rectangle Background
    :param img: Image to put text rect on
    :param text: Text inside the rect
    :param pos: Starting position of the rect x1,y1
    :param scale: Scale of the text
    :param thickness: Thickness of the text
    :param colorT: Color of the Text
    :param colorR: Color of the Rectangle
    :param font: Font used. Must be cv2.FONT....
    :param offset: Clearance around the text
    :param border: Outline around the rect
    :param colorB: Color of the outline
    :return: image, rect (x1,y1,x2,y2)
    """
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)

    return img, [x1, y2, x2, y1]



# Loop
while True:
    success, img = cap.read()
    hands = detector.findHands2(img, draw=False)

    if hands:

        #lmList = hands[0]
        # Dimensions of the bbox
        #lmList, bbox= detector.findPosition(img)
        #print(find_hand_position)
        #print(bbox)
        # List of x and y landmarks positions for all the 21 points
        #lm_list = hands[0]
        lm_list, bbox = detector.findPosition(img)
        x, y, w, h = bbox
        #print(lm_list[5][1:])
        # x and y position for landmk 5 and 17
        x1, y1 = lm_list[5][1:]
        x2, y2 = lm_list[17][1:]
        # find distance between landmks 5 and 17 as a hypotenuse
        # Solving the rotation problem of the distance
        distance = int(math.sqrt((abs(x2-x1))**2 + (abs(y2-y1))**2))
        A, B, C = coff
        distanceCM = A*distance**2 + B*distance + C
        # print(distanceCM, distance)

        cv2.rectangle(img, (x, y), (w+10, h+10), (255, 0, 255), 3)
        putTextRect(img, f'{int(distanceCM)} cm', (x+5, y-10))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
















