import os
import HandTrackingModule as htm
import cv2


cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(3, 640)
cap.set(4, 480) # cap.set(4, 480)

imgBackground = cv2.imread("Resources/Background6.png")


# Importing Modes Images to a list
folderPathModes = "Resources/Modes"
# print(os.listdir(folderPathModes))
listImgModesPath = os.listdir(folderPathModes)  # Path
listImgModes = []
for imgMode in listImgModesPath:
    listImgModes.append(cv2.imread(os.path.join(folderPathModes, imgMode)))
print(listImgModes)



# Importing the Icon Images to a list
folderPathIcons = "Resources/Icons"
# print(os.listdir(folderPathModes))
listImgIconsPath = os.listdir(folderPathIcons)  # Path
listImgIcons = []
for imgIconsPath in listImgIconsPath:
    listImgIcons.append(cv2.imread(os.path.join(folderPathIcons, imgIconsPath)))







modeType = 0  # Changing selection mode
selection = -1
counter = 0
selectionSpeed = 13
detector = htm.HandDetector(detectionCon=0.8, maxHands=2)
center_option_1 = (1136, 196)
center_option_2 = (1000, 384)
center_option_3 = (1136, 581)
modePositions = [center_option_1, center_option_2, center_option_3]
counterPause = 0
selectionList = [-1, -1, -1]


while True:
    success, img = cap.read()

    # Find the hand and its landmarks
    #hands, img = detector.findHands(img)  # Putting the hand tracking over the images

    # Overlaying the webcam feed on the background image
    imgBackground[139:139+480, 47: 47+640] = img  # imgBackground[139:139+360, 51: 51+640] = img
    imgBackground[0:720, 847: 1280] = listImgModes[modeType]  # Adding the first image mode over the image


    # Find the hand and its landmarks
    hands, img = detector.findHands(img)   # Putting the hand tracking behind the images
    if hands and counterPause == 0 and modeType < 3:
        # Hand 1
        hand1 = hands[0]
        fingers1 = detector.fingersUp(hand1)  # How many fingers are up
        print(fingers1)


        # Selecting with fingers up

        # 1 finger
        if fingers1 == [0, 1, 0, 0, 0]:  # Up the index finger (1 finger)
            if selection != 1:
                counter = 1
            selection = 1
        # 2 fingers
        elif fingers1 == [0, 1, 1, 0, 0]:  # Up the index and middle finger (2 fingers)
            if selection != 2:
                counter = 1
            selection = 2
        # 3 fingers
        elif fingers1 == [0, 1, 1, 1, 0]:  # Up three fingers (3 fingers)
            if selection != 3:
                counter = 1
            selection = 3
        # 3 fingers
        elif fingers1 == [0, 0, 1, 1, 1]:  # Up three fingers (3 fingers)
            if selection != 3:
                counter = 1
            selection = 3
        # 3 fingers
        elif fingers1 == [1, 1, 1, 0, 0]:  # Up three fingers (3 fingers)
            if selection != 3:
                counter = 1
            selection = 3
        # 3 fingers
        elif fingers1 == [1, 1, 0, 0, 1]:  # Up three fingers (3 fingers)
            if selection != 3:
                counter = 1
            selection = 3

        else:
            selection = -1
            counter = 0

        if counter > 0:
            counter += 1
            print(counter)

            cv2.ellipse(imgBackground, modePositions[selection -1], (103, 103), 0, 0, counter*selectionSpeed, (0, 255, 0), 20 )

            # Changing Mode
            if counter*selectionSpeed > 360:
                selectionList[modeType] = selection
                modeType+=1
                counter = 0
                selection = -1
                counterPause = 1

    # Making the pause after each selection is made
    if counterPause > 0:
        counterPause += 1
        if counterPause > 40:
            counterPause = 0

    # Add selection Icon at the bottom
    if selectionList[0] != -1:
        imgBackground[650:650+65, 93:93+65] = listImgIcons[selectionList[0] - 1]  # imgBackground[636:636+65, 133:133+65] = listImgIcons[selectionList[0] - 1]
    if selectionList[1] != -1:
        imgBackground[650:650+65, 330:330+65] = listImgIcons[2+selectionList[1]]  # imgBackground[636:636+65, 340:340+65] = listImgIcons[2+selectionList[1]]
    if selectionList[2] != -1:
        imgBackground[650:650+65, 556:556+65] = listImgIcons[5+selectionList[2]]  # imgBackground[636:636+65, 542:542+65] = listImgIcons[5+selectionList[2]]

    # Displaying the image
    #cv2.imshow("Image", img)
    cv2.imshow("Background", imgBackground)
    cv2.waitKey(1)

cv2.destroyAllWindows()

