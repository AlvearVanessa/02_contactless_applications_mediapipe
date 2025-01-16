""""
Mouse Controller - General distance, cameras and screen
@uthor: maalvear
Sources:
https://www.computervision.zone/courses/ai-virtual-mouse/
https://learnopencv.com/center-stage-for-zoom-calls-using-mediapipe/
Creation date:
Wed 11/12/24 15:54 CET
Last modification:
Thu 12/12/24 10:14 CET
"""

# Import libraries
import mediapipe as mp
import cv2
import numpy as np
# import HandTrackingModule_general_zoom as htm
import helper_functions as hf
import time
import autopy
import mouse
# from face_mesh import FaceMeshDetector
# import put_text_rectangle
from vidstab import VidStab

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
threshold_scroll_up = 95 # 50
threshold_scroll_down = 70
smoothing = 5

# Define screen size
w_screen, h_screen = autopy.screen.size()
print("w_screen, h_screen", w_screen, h_screen)

# Caption camera in real-time
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Detect 1 hand (closer to the camera)
detector = hf.HandDetector(max_hands=1)

# Detect 1 face (closer to the camera)
detector_face = hf.FaceMeshDetector(maxFaces=1)

stabilizer = VidStab()

# Define loop to interact with mouse functions using the hand detected
while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = hf.frame_manipulate(img)
    # Stabilize the image to make sure that the changes with Zoom are very smooth
    img = stabilizer.stabilize_frame(input_frame=img,
                                     smoothing_window=2, border_size=-20)
    # Resize the image to make sure it does not crash cam
    img = cv2.resize(img, (800, 480),
                     interpolation=cv2.INTER_CUBIC)  # (800, 480) (int(1280//3), int(720//3))
    img, faces = detector_face.findFaceMesh(img, draw=True)
    # Define height and width of the camera based on image shape (camera size)
    h_cam, w_cam, _ = img.shape
    print("camera resolution h_cam, w_cam = ", h_cam, w_cam)
    A = w_cam - 6*frame_reduction  # 3*frame_reduction - 20  #  3*frame_reduction - 20
    B = w_cam - 2*frame_reduction
    C = h_cam - 4*frame_reduction
    D = h_cam - frame_reduction

    # 2. Define the interactions:
    # move, scroll-up, scroll-down, click
    if success:
        hands, img = detector.find_hands(img)  # with draw
        lm_list, bbox = detector.find_position(img)

        if hands:
            # 3. Select 1 hand
            hand1 = hands[0]
            # lm_list1 = hand1["lm_list"]  # List of 21 Landmark points in x, y, and z pixel position
            fingers1 = detector.fingers_up(hand1)
            # 4. Select x and y pixel position od landmarks 8 and 12 (index and middle tips)
            if len(lm_list) != 0:
                x1, y1 = lm_list[8][1:]

                # 5. Only Index Finger : Moving Mode, if index up and middle finger is down
                if fingers1[1] == 1 and fingers1[2] == 0:
                    # 6. Convert Coordinates to resize screen and makes the move short in real, long in screen
                    # x3 = np.interp(x1, (frame_reduction, w_cam-frame_reduction), (0, w_screen))
                    # y3 = np.interp(y1, (frame_reduction, h_cam-frame_reduction), (0, h_screen))
                    x3 = np.interp(x1, (A, B), (0, w_screen))
                    y3 = np.interp(y1, (C, D), (0, h_screen))

                    # 7. Smoothen Values (to avoid the mouse checking)
                    cloc_x = ploc_x + (x3 - ploc_x) / smoothing
                    cloc_y = ploc_y + (y3 - ploc_y) / smoothing

                    # 8. Move Mouse
                    x_move_screen = w_screen - cloc_x
                    y_move_screen = cloc_y
                    time.sleep(0.002) # time.sleep(0.03)
                    autopy.mouse.move(x_move_screen, cloc_y)
                    cv2.circle(img, (x1, y1), 20, NEON_GREEN_COLOR, cv2.FILLED)
                    ploc_x, ploc_y = cloc_x, cloc_y

            # 9. Find distance between fingers landmark 12 and 0
            length, _, lineInfo = detector.find_distance(12, 0, img)
            # print("line info 12 and 0 = ", length)

            # 10. Clicking Mode if distance greater than a value and all fingers are up
            if fingers1[0] == 1 and fingers1[1] == 1 and fingers1[2] == 1 and fingers1[3] == 1 and fingers1[4] == 1:
                if length > 33: # Allow move continuously
                    # 11. Find distance between fingers landmark 12 and 0
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                               15, NEON_GREEN_COLOR, cv2.FILLED)
                    autopy.mouse.click()
                    time.sleep(0.05)  # 0.75

            # 12. Mouse scrolling bottom up
            if fingers1[1] == 1 and fingers1[2] == 1 and fingers1[3] == 0:
                time.sleep(0.02)  # time.sleep(0.05)
                # 13. Find distance between fingers
                length2, img, lineInfo2 = detector.find_distance(8, 12, img)
                # print("distance tips 8, 12 = ", length2)
                # 14. Clock mouse if distance short
                if length2 < threshold_scroll_up:
                    cv2.circle(img, (lineInfo2[4], lineInfo2[5]),
                               15, BLUE_COLOR, cv2.FILLED)
                    mouse.wheel(delta=1)

            # 15. Mouse scrolling bottom DOWN
            if fingers1[1] == 0 and fingers1[2] == 0:
                time.sleep(0.02)  # tim e.sleep(0.07)
                # 16. Find distance between fingers
                length3, img, lineInfo3 = detector.find_distance(8, 12, img)
                # print("length3 lmks 8 y 12 = ", length3)
                # 17. Clock mouse if distance short
                if length3 < threshold_scroll_down:
                    cv2.circle(img, (lineInfo3[4], lineInfo3[5]),
                               15, BLUE_COLOR, cv2.FILLED)
                    mouse.wheel(delta=-1)

        if faces:
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]
            # Calculate distance
            w, _ = detector_face.findDistance(pointLeft, pointRight)
            # print("distance between pupils = ", w)
            W = 6.3

            # # Finding the Focal Length
            # d = 50
            # f = (w*d)/W
            # print(f)

            # Finding distance
            f = 840
            d = (W * f) / w
            #print(d)
            hf.putTextRect(img, f'Depth: {int(d) - 22}cm', (face[10][0] - 100, face[10][1] - 50),
                                           scale=2)

        # Magenta rectangle where the image is resized
        cv2.rectangle(img, (A, C), (B, D),  MAGENTA_COLOR, 4)

        # 20. Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - p_time)
        p_time = cTime

        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    BLUE_COLOR, 3)

        # 21. Display
        cv2.namedWindow("Big Screen")  # Create a named window
        cv2.setWindowProperty("Big Screen", cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow("Big Screen", int((w_screen-2*w_cam)/2)+400, int((h_screen-2*h_cam)/2)+200)  # Move it to (x,y) (0, 110)
        img_flip = cv2.flip(img, 1)
        cv2.imshow("Big Screen", img_flip)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
