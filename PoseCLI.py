import argparse
import logging
from poserecognition import poserecognition



import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

def main():
    logging.basicConfig(
        # Define logging level
        level=logging.INFO,)

    mode, test = buildArgParser()
    logging.info("Starting pose recognition CLI on mode " + str(mode) + " test " + str(test))
    test = True

    # Start loop
    if mode == 'p':
        pr = poserecognition(test)
        pr.loopRecognition()
    elif mode == 'a':
        pr = poserecognition(test)
        pr.loopAiming(test)
    elif mode == 'f':
        base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)
        pr = poserecognition(test)
        pr.loopAimingFace(test, detector)
    elif mode == 's':
        pr = poserecognition(test,enableSegmentation=True)
        pr.loopSegmentation()


def buildArgParser():

    # Parser instance
    parser = argparse.ArgumentParser()
    # Add allowed arguments
    parser.add_argument("-m", default='f', nargs='?', choices=['p', 'a', 'f', 's'],
                        help="Pose recognition (p) | Aiming - body (a) | Aiming - face (f) | Segmentation (s). For pose recognition mode, hit 'p' to toggle 3D")
    parser.add_argument("-t", action='store_true', help="Enable test mode: no command sent")

    args = parser.parse_args()

    return args.m, args.t


if __name__ == '__main__':
    exit(main())
