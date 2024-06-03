"""Simple standalone testing script for face recognition. Not used in the project"""

import cv2
import mediapipe as mp

class multifaceDetector(object):

    # mediapipe drawing utils
    m_mp_drawing = mp.solutions.drawing_utils
    # load face detection model
    m_mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=0,  # model selection
        min_detection_confidence=0.6  # confidence threshold
    )

    def __init__(self):
        pass

    def detectFaces(self, img):
        # convert color scheme to mediapipe
        img_input = img
        #img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # get results
        results = self.m_mp_face.process(img_input)
        num_faces = 0

        if not results.detections:
            print('No faces detected.')
        else:
            num_faces = len(results.detections)
            print("Detected " + str(num_faces) + " faces")
            for detection in results.detections:  # iterate over each detection and draw on image
                self.m_mp_drawing.draw_detection(img, detection)
        cv2.imshow('multiface', img_input)
        return num_faces

cam = cv2.VideoCapture(0)
faceDetect = multifaceDetector()

while True:
    _, myf = cam.read()
    faceDetect.detectFaces(myf)
    cv2.waitKey(1)