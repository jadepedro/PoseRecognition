import argparse
import logging
from poserecognition import poserecognition


def main():
    logging.basicConfig(
        # Define logging level
        level=logging.INFO,)

    mode, test = buildArgParser()
    logging.info("Starting pose recognition CLI on mode " + str(mode) + " test " + str(test))

    # Start loop
    pr = poserecognition(test)
    if mode == 'p':
        pr.loopRecognition()
    elif mode == 'a':
        pr.loopAiming()


def buildArgParser():

    # Parser instance
    parser = argparse.ArgumentParser()
    # Add allowed arguments
    parser.add_argument("-m", default='p', nargs='?', choices=['p', 'a'],
                        help="Pose recognition (p) | Aiming (a)")
    parser.add_argument("-t", action='store_true', help="Enable test mode: no command sent")

    args = parser.parse_args()

    return args.m, args.t


if __name__ == '__main__':
    exit(main())
