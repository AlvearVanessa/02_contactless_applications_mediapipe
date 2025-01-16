"""
Define functions to create a contactless interaction
controlling 4 mouse functionalities
Sources:
https://teachablemachine.withgoogle.com/
https://www.computervision.zone/courses/hand-sign-detection-asl/
https://learnopencv.com/center-stage-for-zoom-calls-using-mediapipe/
https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_detection.py

Mon 18/11/24 12:07 CET
Last modification:
Thu 12/12/2024 10:13 CET
"""

# Import libraries
import cv2
import mediapipe as mp
import math
import numpy as np
import time
from vidstab import VidStab

# global variables
global fingers
gb_zoom = 1.4 # 1.4

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

def zoom_at(image, coord=None, zoom_type=None):
    """
    Args:
        image: frame captured by camera
        coord: coordinates of face(nose)
        zoom_type:Is it a transition or normal zoom
    Returns:
        Image with cropped image
    """
    global gb_zoom
    # If zoom_type is transition check if Zoom is already done else zoom by 0.1 in current frame
    if zoom_type == 'transition' and gb_zoom < 3.0: # 3.0
        gb_zoom = gb_zoom + 0.1  # 0.1

    # If zoom_type is normal zoom check if zoom more than 1.4 if so zoom out by 0.1 in each frame
    if gb_zoom != 1.4 and zoom_type is None: # 1.4
        gb_zoom = gb_zoom - 0.1 # 0.1

    zoom = gb_zoom
    # If coordinates to zoom around are not specified, default to center of the frame.
    cy, cx = [i / 2 for i in image.shape[:-1]] if coord is None else coord[::-1]
    # print("image.shape[:-1], coord[::-1] = ",image.shape[:-1], coord[::-1])

    # Scaling the image using getRotationMatrix2D to appropriate zoom.
    rot_mat = cv2.getRotationMatrix2D((cx, cy), 0, zoom)

    # Use warpAffine to make sure that  lines remain parallel
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def frame_manipulate(img):
    """
    Args:
        img: frame captured by camera
    Returns:
        Image with manipulated output
    """
    # Mediapipe face set up
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Set the default values to None
        coordinates = None
        zoom_transition = None
        if results.detections: # "detections" field that contains a list of the detected face location data.
            for detection in results.detections:
                height, width, channels = img.shape

                # Fetch coordinates of nose, right ear and left ear
                nose = detection.location_data.relative_keypoints[2]
                right_ear = detection.location_data.relative_keypoints[4]
                left_ear = detection.location_data.relative_keypoints[5]

                #  get coordinates for right ear and left ear
                right_ear_x = int(right_ear.x * width)
                left_ear_x = int(left_ear.x * width)

                # Fetch coordinates for the nose and set as center
                center_x = int(nose.x * width)
                center_y = int(nose.y * height)
                coordinates = [center_x, center_y]
                print("distance in pixels left and right ear:", left_ear_x - right_ear_x)
                # Check the distance between left ear and right ear if distance is less than 120 pixels zoom in
                if (left_ear_x - right_ear_x) < 80: #120
                    zoom_transition = 'transition'

        # Perform zoom on the image
        img = zoom_at(img, coord=coordinates, zoom_type=zoom_transition)

    return img




def main():
    # Video Stabilizer
    # device_val = None
    stabilizer = VidStab()

    # For webcam input:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensions to cam object (not cap) 800
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # 480
    cap.set(cv2.CAP_PROP_FPS, 120)

    # Check OS
#    os = platform.system()
#    if os == "Linux":
#        device_val = "/dev/video2"

    # Start virtual camera
#    with pyvirtualcam.Camera(1280, 720, 120, device=device_val, fmt=PixelFormat.BGR) as cam:
#        print('Virtual camera device: ' + cam.device)

    while True:
        success, img = cap.read()
        img = frame_manipulate(img)
        # Stabilize the image to make sure that the changes with Zoom are very smooth
        img = stabilizer.stabilize_frame(input_frame=img,
                                         smoothing_window=2, border_size=-20)
        # Resize the image to make sure it does not crash cam
        img = cv2.resize(img, (1280, 720),
                         interpolation=cv2.INTER_CUBIC)  # (800, 480) (int(1280//3), int(720//3))

        # cam.send(img)
        # cam.sleep_until_next_frame()
        # Display
        img_flip = cv2.flip(img, 1)
        cv2.imshow("Image", img_flip)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == '__main__':
    main()


class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, max_hands=2, detection_con=0.5, min_track_con=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param max_hands: Maximum number of hands to detect
        :param detection_con: Minimum Detection Confidence Threshold
        :param min_track_con: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.min_track_con = min_track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode, max_num_hands=self.max_hands,
                                         min_detection_confidence=self.detection_con,
                                         min_tracking_confidence=self.min_track_con)
        self.results = self.hands.process
        self.mp_draw = mp.solutions.drawing_utils
        # To select the x and y position of certain points hand_landmarks.landmark[4:21:4]
        self.tip_ids = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lm_list = []

    def find_hands(self, img, draw=True, flip_type=True):
        """
        Find hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
        self.results = self.hands.process(img)
        all_hands = []
        h, w, c = img.shape
        # print("h_, w_ = ", h_, w_)
        # black = np.zeros((400, 470, 3), dtype=np.uint8)
        black = np.zeros((h, w, c), dtype=np.uint8)  #np.zeros((1640, 3060, 3), dtype=np.uint8)
        annotated_image = black
        if self.results.multi_hand_landmarks:
            for hand_type, hand_lms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                my_hand = {}
                # lm_list
                my_lm_list = []
                x_list = []
                y_list = []
                for id_, lm in enumerate(hand_lms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    my_lm_list.append([px, py, pz])
                    x_list.append(px)
                    y_list.append(py)

                # bbox
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                box_w, box_h = x_max - x_min, y_max - y_min
                bbox = x_min, y_min, box_w, box_h
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                my_hand["lm_list"] = my_lm_list
                my_hand["bbox"] = bbox
                my_hand["center"] = (cx, cy)

                if flip_type:
                    if hand_type.classification[0].label == "Right":
                        my_hand["type"] = "Left"
                    else:
                        my_hand["type"] = "Right"
                else:
                    my_hand["type"] = hand_type.classification[0].label
                all_hands.append(my_hand)

                # drawing landmarks
                hand_found = bool(self.results.multi_hand_landmarks)
                if hand_found:
                    for hand_landmarks in self.results.multi_hand_landmarks:
                        # draw
                        if draw:
                            self.mp_draw.draw_landmarks(annotated_image, hand_landmarks,
                                                        self.mp_hands.HAND_CONNECTIONS)
                            cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                          (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                          LIGHT_PINK_COLOR, 2)
                            cv2.putText(img, my_hand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                        2, MAGENTA_COLOR, 2)

        if draw:
            return all_hands, annotated_image
        else:
            return all_hands, annotated_image

    def fingers_up(self, my_hand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        my_hand_type = my_hand["type"]
        my_lm_list = my_hand["lm_list"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if my_hand_type == "Right":
                if my_lm_list[self.tip_ids[0]][0] > my_lm_list[self.tip_ids[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if my_lm_list[self.tip_ids[0]][0] < my_lm_list[self.tip_ids[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id_ in range(1, 5):
                if my_lm_list[self.tip_ids[id_]][1] < my_lm_list[self.tip_ids[id_] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def find_position(self, img, handNo=0, draw=False):
        x_list = []
        y_list = []
        bbox = []
        self.lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[handNo]
            # print(my_hand)
            for id, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                # cx is the pixel position in image of the x landmark position, same for cy
                # lm.x is a ratio of the pixel position
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                # print(id, lm.x, cx, lm.y, cy)
                self.lm_list.append([id, cx, cy])
                if draw:
                    # cv2.circle(img, (int(lm.x), int(lm.y)), 200, GREEN_COLOR, cv2.FILLED)
                    cv2.circle(img, (cx, cy), 5, MAGENTA_COLOR, cv2.FILLED)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            # bbox = x_min, y_min, x_max, y_max
            bbox.append(x_min)
            bbox.append(y_min)
            bbox.append(x_max)
            bbox.append(y_max)

            if draw:
                cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20),
                              NEON_GREEN_COLOR, 2)

        return self.lm_list, bbox

    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), LIGHT_BLUE_COLOR, t)
            cv2.circle(img, (x1, y1), r, LIGHT_BLUE_COLOR, cv2.FILLED)
            cv2.circle(img, (x2, y2), r, LIGHT_BLUE_COLOR, cv2.FILLED)
            cv2.circle(img, (cx, cy), r, BLUE_COLOR, cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def find_hands_draw(self, img, draw=True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
        # print(results.multi_hand_landmarks)

        # results1 = hand_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        hand_found = bool(self.results.multi_hand_landmarks)
        h, w, c = img.shape
        # print("h, w = ", h, w)

        if hand_found:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    # black = np.zeros((400, 470, 3), dtype=np.uint8)
                    black = np.zeros((h, w, c), dtype=np.uint8)
                    # black = np.zeros((1640, 3060, 3), dtype=np.uint8)
                    annotated_image = black
                    self.mp_draw.draw_landmarks(annotated_image, hand_lms,
                                                self.mp_hands.HAND_CONNECTIONS)

                    return annotated_image

zoom_factor = 1.3
def main():
    p_time = 0
    c_time = 0
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    detector = HandDetector(detection_con=0.5, max_hands=1)
    while True:
        # Get image frame
        success, img = cap.read()
        img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
        # h_cap, w_cap, _ = img.shape
        # print("h_cap, w_cap", h_cap, w_cap)

        # Find the hand and its landmarks
        hands, img = detector.find_hands(img)  # with draw
        # hands = detector.findHands(img, draw=False)  # without draw
        lm_list, bbox = detector.find_position(img)
        # print(bbox)

        # Find Distance between two Landmarks. Could be same hand or different hands
        length, img, info = detector.find_distance(0, 0, img)  # with draw
        # print(length)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    MAGENTA_COLOR, 3)

        # Display
        img_flip = cv2.flip(img, 1)
        cv2.imshow("Image", img_flip)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()





class FaceMeshDetector:
    """
    Face Mesh Detector to find 468 Landmarks using the mediapipe library.
    Helps acquire the landmark points in pixel format
    """

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        """
        :param staticMode: In static mode, detection is done on each image: slower
        :param maxFaces: Maximum number of faces to detect
        :param minDetectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,
                                                 max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        """
        Finds face landmarks in BGR Image.
        :param img: Image to find the face landmarks in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print("shape", self.imgRGB.shape)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces

    def findDistance(self,p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length,info, img
        else:
            return length, info


def main():
    # Initialize the webcam
    # '2' indicates the third camera connected to the computer, '0' would usually refer to the built-in webcam
    cap = cv2.VideoCapture(2 , cv2.CAP_DSHOW)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # h_cap, w_cap, _ = img.shape
    # print("height, width", height, width)
    # Initialize FaceMeshDetector object
    # staticMode: If True, the detection happens only once, else every frame
    # maxFaces: Maximum number of faces to detect

    # minDetectionCon: Minimum detection confidence threshold
    # minTrackCon: Minimum tracking confidence threshold
    detector = FaceMeshDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)

    # Start the loop to continually get frames from the webcam
    while True:
        # Read the current frame from the webcam
        # success: Boolean, whether the frame was successfully grabbed
        # img: The current frame
        success, img = cap.read()


        # Find face mesh in the image
        # img: Updated image with the face mesh if draw=True
        # faces: Detected face information
        img, faces = detector.findFaceMesh(img, draw=True)

        # Check if any faces are detected
        if faces:
            # Loop through each detected face
            for face in faces:
                # Get specific points for the eye
                # leftEyeUpPoint: Point above the left eye
                # leftEyeDownPoint: Point below the left eye
                leftEyeUpPoint = face[159]
                leftEyeDownPoint = face[23]

                # Calculate the vertical distance between the eye points
                # leftEyeVerticalDistance: Distance between points above and below the left eye
                # info: Additional information (like coordinates)
                leftEyeVerticalDistance, info = detector.findDistance(leftEyeUpPoint, leftEyeDownPoint)

                # Print the vertical distance for debugging or information
                # print(leftEyeVerticalDistance)

        # Display the image in a window named 'Image'
        cv2.imshow("Image", img)

        # Wait for 1 millisecond to check for any user input, keeping the window open
        cv2.waitKey(1)


if __name__ == "__main__":
    main()


