import cv2
import time
import os
import HandTrackingModule as htm

#Parameters for camera size
wCam, hCam = 640, 480

cap = cv2.VideoCapture(1)
# Setting sizes adn 3 and 4 are Id's
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
# Store finger images
myList = os.listdir(folderPath)
#print(myList)

# Create a list of images
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

#print(len(overlayList))
pTime = 0


detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)





    if len(lmList) != 0:
        fingers = []


        # Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:  # it means this is open, only 1 landmark below and the ID is zero
            fingers.append(1)
        else:
            fingers.append(0)


        # 4 fingers
        for id in range(1, 5):
            # Try to get tip of our fingers
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:  # it means this is open
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)

        # Total fingers
        totalFingers = fingers.count(1)  # How many !s do we have
        print(totalFingers)




        #  For changing the image
        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]  # for an image with size h and w- Change the 0 Value for one of the totalFingers

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                10, (255, 0, 0), 25)  # Magenta color (255, 0, 255)


    # Adding frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    cv2.putText(img, f'FPS:{int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


