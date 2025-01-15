import cv2
import HandTrackingModule as htm
import socket

# Parameters
width, height = 1280, 720

# Webcam setting
cap = cv2.VideoCapture(0)
# Big area to move the hand
cap.set(3, width)  # width
cap.set(4, height)  # height

# Hand detector
detector = htm.HandDetector(maxHands=1, detectionCon=0.8)

# Create the communication with Unity
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

while True:
    # Get the frame from the webcam
    success, img = cap.read()

    # Hands
    hands, img = detector.findHands(img)

    data = []
    # Landmark values  - (x,y,z)*21
    if hands:
        # Get the first hand detected
        hand = hands[0]
        # Get the landmark list

        # We obtain [[491, 747, 0], [561, 704, -41],...,[644, 330, -50]] and we want to
        # remove the brackets [491, 747, 0], [561, 704, -41],... to get one single list
        # hand is a list, but when we put the hand['lmList'] they will give us a list or our lm
        lmList = hand['lmList']
        # print(lmList)

        for lm in lmList:
            data.extend([lm[0], height - lm[1], lm[2]]) # For changing the second (y) component height - lm[1]
        # print(data)  # [759, 739, 0, 822, 693, -9, 862, ..., 634] with data.extend(lm)
        sock.sendto(str.encode(str(data)), serverAddressPort)


    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    #img = cv2.flip(img, 1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
