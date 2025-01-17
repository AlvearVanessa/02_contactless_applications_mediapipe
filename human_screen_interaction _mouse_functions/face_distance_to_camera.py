"""
20 03 2024 13:51h CET
Face Mesh Module
Source: https://www.computervision.zone/
"""


import cv2
from face_mesh import FaceMeshDetector

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
detector = FaceMeshDetector(maxFaces=1)

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


while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        face = faces[0]
        # Select the landmark 145 (right eye) for the face 0
        pointRight = face[145]
        # Select the landmark 347 (left eye) for the face 0
        pointLeft = face[374]
        # Drawing
        #cv2.line(img, pointLeft, pointRight, 3)
        #cv2.circle(img, pointLeft, 5, GREEN_COLOR, cv2.FILLED)
        #cv2.circle(img, pointRight, 5, BLUE_COLOR, cv2.FILLED)
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3

        # # Finding the Focal Length
        # d = 50
        # f = (w*d)/W
        # print(f)

        # Finding distance
        f = 840
        d = (W * f) / w
        print(d)

        putTextRect(img, f'Depth: {int(d)-22}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)

    cv2.namedWindow("Distancia persona - camara")  # Create a named window
    cv2.setWindowProperty("Distancia persona - camara", cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow("Distancia persona - camara", img)
    cv2.waitKey(1)
