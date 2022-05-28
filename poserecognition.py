import logging
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

# use the custom drawing utils for plotting the landmarks in 3D
from mediapipedrawing_utils import plot_landmarks
from videoloop import videoloop
from arduinoControl import arduinoControl


class poserecognition(object):
    """
    Processes a video entry to search for pose recognition
    """

    # drawing tools
    m_mp_drawing = None
    # drawing styles
    m_mp_drawing_styles = None
    # pose recognizer
    m_mp_pose = None
    # 3d graph axes
    m_axes = None
    # figure
    m_fig = None
    # panel
    m_panel = None
    # canvas
    m_canvas = None

    # aiming
    m_arduinoControl = None

    # previous mask for Segmentation
    m_prev_mask = None
    # tick for segmentation
    m_segmentation_tick = 0
    # previous trail
    m_previous_trail = None

    # debug mode
    m_test = False
    # show 3d landmarks flag
    m_show_3d = False

    # colors (BGR)
    MASK_COLOR = (0, 112, 255)

    #################################################
    # Common section                                #
    #################################################
    def __init__(self, test=False, enableSegmentation=False) -> None:
        # test mode
        self.m_test = test
        # Pose estimator drawing tool
        self.m_mp_drawing = mp.solutions.drawing_utils
        # Pose drawing styles
        self.m_mp_drawing_styles = mp.solutions.drawing_styles
        # Pose recognizer
        self.m_mp_pose = mp.solutions.pose
        self.m_pose = self.m_mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=enableSegmentation)

    #################################################
    # Aiming section                                #
    #################################################
    def __aiming(self, frame):
        """
        Performs aiming of a laser connected to an arduino
        :param frame: frame on which perform pose recognition and used as data for aiming
        :return: processed frame and results
        """
        # convert the frame to RGB format
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process the RGB frame to get the result
        results = self.m_pose.process(RGBframe)
        logging.debug(results.pose_landmarks)

        # draw detected 2D skeleton on the frame
        self.m_mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.m_mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.m_mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # deduct angle
        h_angle, v_angle = self.__deductAngle(results)

        # print angle on frame
        cv2.putText(frame, text=str(h_angle), org=(20, 80), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3,
                    color=(0, 255, 0), thickness=3)

        # send angle to arduino
        self.m_arduinoControl.sendServoAngle(h_angle, v_angle)

        return frame

    def __deductAngle(self, results):
        """
        Deduct angle for both H and V axis from landmarks. Review https://colab.research.google.com/drive/1uCuA6We9T5r0WljspEHWPHXCT_2bMKUy#scrollTo=BAivyQ_xOtFp
        :param results: results obtained during pose estimation
        :return: estimated H and V angle
        """
        h_angle = 90
        v_angle = 90
        if not results.pose_landmarks is None:
            # H angle
            # estimate radius: use landmarks for nose and ears
            nose = results.pose_landmarks.landmark[self.m_mp_pose.PoseLandmark.NOSE]
            left_ear = results.pose_landmarks.landmark[self.m_mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = results.pose_landmarks.landmark[self.m_mp_pose.PoseLandmark.RIGHT_EAR]
            # take as radius the difference between left and right x coordinate halfed
            radius = float((left_ear.x - right_ear.x) / 2)
            # take as proyection point the distance between nose to center of ears
            center = right_ear.x + radius
            proy = float(nose.x - center)
            # calculate ratio between proyection and radius
            ratio = proy / radius
            if ratio > 1:
                ratio = 1
            elif ratio < -1:
                ratio = -1
            # calculate estimated angle (in radians)
            h_angle = np.arcsin([ratio])[0]
            # convert to degrees
            h_angle = int(h_angle * 180 / np.pi)

            logging.debug("ratio: " + str(ratio) + " proyection: " + str(proy))

        return h_angle, v_angle

    def loopAiming(self):
        """
        Runs the main vido processing loop for aiming
        :return:
        """
        logging.info("Entering aiming")

        # instantiate arduino controller
        self.m_test = True  # TODO: remove
        self.m_arduinoControl = arduinoControl(self.m_test)

        try:
            # Create a video looper, that uses the aiming function as the frame processing function
            vl = videoloop(self.__aiming)
            # Invoke the main oop
            vl.loopVideoProcess()
        except ValueError:
            logging.error("Error" + str(ValueError))
        logging.info("Exiting aiming")

    #################################################
    # Segmentation section                          #
    #################################################
    def __getMask(self, frame, smoothedMask=True):
        """
        Calculates the segmentation for a given frame
        :param frame: frame on which operated
        :param smoothedMask: if true, it will return smooth borders, if false, sharp ones
        :return: generated mask
        """
        # convert the frame to RGB format
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process the RGB frame to get the result
        results = self.m_pose.process(RGBframe)

        # if no segmentation, just use a black mask
        # changing from float32 to uint8 has the effect of 'smoothing mask'
        mask = np.zeros(frame.shape, dtype=np.uint8)

        # extract segmentation mask
        if results.segmentation_mask is not None:
            # build mask from results
            mask = results.segmentation_mask
            # as segmentation returns values within 0 to 1, multiply every pixel and channel by 255 to show
            mask = mask * 255
            # process smoothed mask
            # changing from float32 to uint8 has the effect of 'smoothing mask'
            if smoothedMask:
                mask = mask.astype(np.uint8)
        else:
            # if no segmentation, just use a black mask. Mask are always grey scale
            mask = np.zeros(frame.shape, dtype=np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        return mask

    def __getTrail(self, frame, mask, delay=2):
        """
        Calculates the trail between current and previous masks
        Only for a given number of ticks
        :param mask: current mask
        :param delay: delay between effectively applied trails
        :return: applied flag, trailed image
        """
        applied = False
        trail = None

        if (self.m_segmentation_tick % delay) == 0:
            # generate solid trail for previous mask
            solid_mask = np.zeros(frame.shape, dtype=np.uint8)
            solid_mask[:] = self.MASK_COLOR
            #cv2.imshow('solid mask', solid_mask)
            trail = cv2.bitwise_and(solid_mask, solid_mask, mask=self.m_prev_mask)

            # update previous mask and trail
            self.m_prev_mask = mask
            self.m_previous_trail = trail
            applied = True
        self.m_segmentation_tick += 1

        return applied, self.m_previous_trail

    def __composeForeAndBackgroud(self, fg_image, bg_image, mask):
        """
        Composes two images accordingly to mask
        :param fg_image: the forewround image
        :param bg_image: the background image
        :param mask: mask to compose
        :return: composed image
        """

        condition = np.stack(
            (mask,) * 3, axis=-1) > 0.1
        output_image = np.where(condition, fg_image, bg_image)

        return output_image

    def __overallMask(self,composed_image):
        """
        Calculates overall mask including current segmentation and trail
        :param composed_image: current composed image including trails
        :return: full mask including current and previous
        """
        # convert to HSV
        imgHSV = cv2.cvtColor(composed_image, cv2.COLOR_BGR2HSV)
        # create mask with only black pixels
        mask = cv2.inRange(imgHSV, np.array([0,0,0]), np.array([0,0,0]))
        mask = cv2.bitwise_not(mask)

        #cv2.imshow('overall mask', mask)
        return mask

    def __poseSegmentation(self, frame):

        try:
            # extract mask
            mask = self.__getMask(frame, smoothedMask=True)
            # show mask
            cv2.imshow('mask', mask)

            # show masked image
            # calculate the matching area
            masked = cv2.bitwise_and(frame, frame, mask=mask)
            #cv2.imshow('masked', masked)

            # if previous mask is None, initialize to black
            if self.m_prev_mask is None:
                self.m_prev_mask = mask
                # self.m_prev_mask = np.zeros(frame.shape, dtype=np.uint8)
            # calculate trail
            applied, trail = self.__getTrail(frame, mask)
            # compose frame and trail
            trailed = self.__composeForeAndBackgroud(frame, trail, mask)

            # calculate overall mask
            overall_mask = self.__overallMask(trailed)

            # compose trailed image with background
            final_composition = self.__composeForeAndBackgroud(trailed, frame, overall_mask)
        except ValueError:
            pass


        # show trailed image
        #if applied:
        #    cv2.imshow('trail', trail)
        cv2.imshow('final composition', final_composition)

        return frame

    def loopSegmentation(self):
        """
        Runs the main vido processing loop for segmentation
        :return:
        """
        logging.info("Entering segmentation")

        try:
            # Create a video looper, that uses the aiming function as the frame processing function
            vl = videoloop(self.__poseSegmentation)
            # Invoke the main oop
            vl.loopVideoProcess()
        except ValueError:
            logging.error("Error" + str(ValueError))
        logging.info("Exiting segmentation")

    #################################################
    # Pose recognition section                      #
    #################################################

    def __toggle3Dview(self):
        """
        Shows the 3D proyection
        :return:
        """
        # set matplotlib drawing assets
        self.m_axes = plt.axes(projection='3d')
        self.m_fig = self.m_axes.figure
        self.m_canvas = self.m_fig.canvas
        self.m_axes.figure.canvas.mpl_connect('close_event', self.__on_close)
        self.m_canvas.set_window_title('3D estimation')

        # view default values
        angle = 30
        elevation = 10
        azimuth = 10
        self.m_axes.view_init(elev=elevation, azim=azimuth)
        self.m_axes.view_init(angle, 90 - angle)
        self.m_show_3d = True
        logging.debug("toggle visibility: " + str(self.m_show_3d))

    def __on_close(self, event):
        """
        Event handler for closing the 3D window
        :param event:
        :return:
        """
        logging.debug("closing figures")
        self.m_show_3d = False

    def __poseRecognition(self, frame):
        """
        Performs pose recognition in a given frame. Optionally shows 3D estimation
        :param frame: frame on which perform the pose recognition
        :return: processed frame and results
        """
        # convert the frame to RGB format
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process the RGB frame to get the result
        results = self.m_pose.process(RGBframe)
        logging.debug(results.pose_landmarks)

        # draw detected 2D skeleton on the frame
        self.m_mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.m_mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.m_mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # plot 3D landmarks
        if self.m_show_3d:
            plot_landmarks(
                self.m_axes,
                results.pose_world_landmarks,
                self.m_mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.m_mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # toggle 3D if requested
        if cv2.waitKey(1) & 0xFF == ord('p') and not self.m_show_3d:
            self.__toggle3Dview()

        return frame

    def loopRecognition(self):
        """
        Runs the main video processing loop with the pose recognition
        :return:
        """
        logging.debug("Entering pose recognition")
        try:
            # Create a video looper, that uses the pose recognition function as the frame processing function
            vl = videoloop(self.__poseRecognition)
            # Invoke the main oop
            vl.loopVideoProcess()
        except ValueError:
            logging.error("Error" + str(ValueError))
        logging.info("Exiting pose recognition")
