""""
AI Virtual Mouse - General cameras and screen
Thur 26/09/24 08:43 CET
@uthor: maalvear
Source: https://www.computervision.zone/courses/ai-virtual-mouse/

"""

import cv2
import numpy as np
import HandTrackingModule_general as htm
import time
import autopy
import mouse
from face_mesh import FaceMeshDetector
# import tkinter as tk

# Define color variables regarding OpenCV
WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
LIGHT_PINK_COLOR = (255, 153, 255)
MAGENTA_COLOR = (255, 0, 255)
LIGHT_BLUE_COLOR = (255, 255, 0)
YELLOW_COLOR = (0, 255, 255)
NEON_GREEN_COLOR = (0, 255, 0)
ORANGE_COLOR = (0, 128, 255)
LAVENDER_COLOR = (255, 0, 128)

# Define variables
frame_reduction = 100  # Frame Reduction
p_time = 0
ploc_x, ploc_y = 0, 0
cloc_x, cloc_y = 0, 0
threshold_click = 40
threshold_scroll_up = 25 # 50
threshold_scroll_down = 35
smoothing = 5

# Define screen size
w_screen, h_screen = autopy.screen.size()
print("w_screen, h_screen", w_screen, h_screen)

# Caption of the camera in real-time
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Find 1 hand detected (closer to the camera)
detector = htm.HandDetector(max_hands=1)

detector_face = FaceMeshDetector(maxFaces=1)



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





# Define loop to interact with mouse functions using the hand detected
while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img, faces = detector_face.findFaceMesh(img, draw=False)

    # Define height and width of the camera based on image shape (camera size)
    h_cam, w_cam, _ = img.shape
    print("h_cam, w_cam", h_cam, w_cam)
    A = 4*frame_reduction - 20
    C = 4*frame_reduction - 60
    B = w_cam - frame_reduction # w_cam - 5*frame_reduction
    D = h_cam - 40 # h_cam - 3*frame_reduction

    # 2. Define the interactions: move, scroll-up, scroll-down, click
    if success:
        # img = cv2.resize(img, (B+200, D+200))
        hands, img = detector.find_hands(img)  # with draw
        lm_list, bbox = detector.find_position(img)

        if hands:
            # 3. Select 1 hand
            hand1 = hands[0]
            # lm_list1 = hand1["lm_list"]  # List of 21 Landmark points in x, y, and z pixel position
            fingers1 = detector.fingers_up(hand1)
            # 4. Select x and y pixel position od  landmarks 8 and 12 (index and middle tips)
            if len(lm_list) != 0:
                x1, y1 = lm_list[8][1:]
                x2, y2 = lm_list[12][1:]

                # 5. Only Index Finger : Moving Mode, if index up and middle finger is down
                if fingers1[1] == 1 and fingers1[2] == 0:
                    # 6. Convert Coordinates to resize screen and makes the move short in real, long in screen
                    x3 = np.interp(x1, (A, B), (0, w_screen))
                    y3 = np.interp(y1, (C, D), (0, h_screen))

                    # 7. Smoothen Values (to avoid the mouse checking)
                    cloc_x = ploc_x + (x3 - ploc_x) / smoothing
                    cloc_y = ploc_y + (y3 - ploc_y) / smoothing
                    print("cloc_x, cloc_y", cloc_x, cloc_y)

                    # 8. Move Mouse
                    x_move_screen = w_screen - cloc_x
                    y_move_screen = cloc_y
                    autopy.mouse.move(x_move_screen, cloc_y)
                    # time.sleep(0.002)
                    cv2.circle(img, (x1, y1), 20, NEON_GREEN_COLOR, cv2.FILLED)
                    ploc_x, ploc_y = cloc_x, cloc_y

            # 9. Find distance between fingers landmark 12 and 0
            length, _, lineInfo = detector.find_distance(12, 0, img)
            # print("line info 12 and 0", lineInfo)

            # 10. Clicking Mode if distance greater than a value and all fingers are up
            if fingers1[0] == 1 and fingers1[1] == 1 and fingers1[2] == 1 and fingers1[3] == 1 and fingers1[4] == 1:
                if length > 33:
                    # 11. Find distance between fingers landmark 12 and 0
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                               15, NEON_GREEN_COLOR, cv2.FILLED)
                    autopy.mouse.click()
                    time.sleep(0.75)  # 0.75

            # 12. Mouse scrolling bottom up
            if fingers1[1] == 1 and fingers1[2] == 1 and fingers1[3] == 0:
                time.sleep(0.04)  # time.sleep(0.05)
                # 13. Find distance between fingers
                length2, img, lineInfo2 = detector.find_distance(8, 12, img)
                # 14. Clock mouse if distance short
                if length2 < threshold_scroll_up:
                    cv2.circle(img, (lineInfo2[4], lineInfo2[5]),
                               15, BLUE_COLOR, cv2.FILLED)
                    mouse.wheel(delta=1)

            # 15. Mouse scrolling bottom DOWN
            if fingers1[1] == 0 and fingers1[2] == 0:
                time.sleep(0.05)  # time.sleep(0.07)
                # 16. Find distance between fingers
                length3, img, lineInfo3 = detector.find_distance(8,   12, img)
                print("length3=", length3)
                # 17. Clock mouse if distance short
                if length3 < threshold_scroll_down:
                    cv2.circle(img, (lineInfo3[4], lineInfo3[5]),
                               15, BLUE_COLOR, cv2.FILLED)
                    mouse.wheel(delta=-1)

            # 18. x and y position for landmark 5 and 17
            x1, y1 = lm_list[5][1:]
            x2, y2 = lm_list[17][1:]

        if faces:
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]
            # Drawing


            # cv2.line(img, pointLeft, pointRight, 3)
            # cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
            # cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
            w, _ = detector_face.findDistance(pointLeft, pointRight)
            W = 6.3

            # # Finding the Focal Length
            # d = 50
            # f = (w*d)/W
            # print(f)

            # Finding distance
            f = 840
            d = (W * f) / w
            print(d)

            # 20. Print
            putTextRect(img, f'Depth: {int(d)-22}cm',
                               (face[10][0] - 100, face[10][1] - 50),
                               scale=2)


        # 19. Print rectangle to resize screen (Edge)
        # x3 = np.interp(x1, (frame_reduction, w_cam - frame_reduction), (0, w_screen))
        # y3 = np.interp(y1, (frame_reduction, h_cam - frame_reduction), (0, h_screen))
        cv2.rectangle(img, (A, C), (B, D),
                      MAGENTA_COLOR, 4)

        # 20. Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - p_time)
        p_time = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    BLUE_COLOR, 3)

        # 21. Display
        cv2.namedWindow("Big Screen")  # Create a named window
        cv2.moveWindow("Big Screen", int(w_screen/2), int(h_screen/2))  # Move it to (x,y) (0, 110) int(w_screen/3), int(h_screen/3)
        img_flip = cv2.flip(img, 1)
        cv2.imshow("Big Screen", img_flip)

        # To Stop interaction
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
