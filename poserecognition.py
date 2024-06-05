import logging
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import time

# needed for face landmarks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# use the custom drawing utils for plotting the landmarks in 3D
from Graphics import mediapipedrawing_utils
from arduinoControl import arduinoControl
from Graphics.GraphicsHelper import GraphicsHelper
from videoloop import videoloop
import LandmarkNumberring.LandMarkNumbering as lmn


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
    m_h_angle = 0
    m_v_angle = 0

    # previous mask for Segmentation
    NUMBER_OF_MASK = 4
    m_prev_mask_array = None
    # tick for segmentation
    m_segmentation_tick = 0
    # previous trail (combined)
    m_previous_trail = None

    # debug mode
    m_test = False
    # show 3d landmarks flag
    m_show_3d = False

    # colors (BGR)
    MASK_COLOR = [(0, 112, 255), (50, 132, 205), (100, 152, 155), (150, 172, 105), (200, 192, 55)]

    #################################################
    # Common section                                #
    #################################################
    def __init__(self, test=False, enableSegmentation=False, shape=(480, 640), detector=None) -> None:
        # graphic debugging tool
        self.graphicHelper = None
        # test mode
        self.m_test = test
        # Pose estimator drawing tool
        self.m_mp_drawing = mp.solutions.drawing_utils
        # Pose drawing styles
        self.m_mp_drawing_styles = mp.solutions.drawing_styles
        # Pose recognizer. This is used for the full body pose estimation
        self.m_mp_pose =None
        self.m_pose = None
        
        # Enabling segmentation
        self.m_enableSegmentation = enableSegmentation

        # Face landmarks recognizer. This is used for the face landmarks estimation
        # See https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python?hl=es-419#video
        #base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        #self.m_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        #self.m_options = vision.FaceLandmarkerOptions(base_options=base_options,
        #                                    output_face_blendshapes=False,
        #                                    output_facial_transformation_matrixes=False,
        #                                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
        #                                    num_faces=1)
        #self.m_detector = vision.FaceLandmarker.create_from_options(self.m_options)
        self.m_detector = detector

        # previous trails
        self.m_prev_mask_array = []
        for i in range(self.NUMBER_OF_MASK):
            self.m_prev_mask_array.append(np.zeros(shape, dtype=np.uint8))
            #cv2.imshow('prev mask ' + str(i), self.m_prev_mask_array[i])
            #cv2.waitKey(0)
        # convert to numpy array
        self.m_prev_mask_array = np.array(self.m_prev_mask_array)

    def initBodyPose(self):
        """
        Initializes body pose estimator. This has been separated to avoid conflict with the face detection.
        :return:
        """
        self.m_mp_pose = mp.solutions.pose
        self.m_pose = self.m_mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=self.m_enableSegmentation)
        
    #################################################
    # Aiming section based on face landmarks        #
    #################################################
    def __aimingFace(self, frame):
        """
        Performs aiming of a laser connected to an arduino
        :param frame: frame on which perform pose recognition and used as data for aiming
        :return: processed frame and results
        """
        # convert the frame to RGB format is not needed as it will be handeld by the MediaPipe operation
        RGBframe = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process the RGB frame to get the result
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGBframe)
        face_landmarker_result = self.m_detector.detect(mp_image)
        #face_landmarker_result = self.m_detector.detect_for_video(mp_image,timestamp_ms=0)

        # draw detected 2D skeleton on the frame
        annotated_image = mediapipedrawing_utils.draw_landmarks_on_image(mp_image.numpy_view(), face_landmarker_result, draw_numbers=False)

        # deduct angle
        self.m_h_angle, self.m_v_angle = self.__deductAngleFace(face_landmarker_result)

        # record new angle on graphic helper
        self.graphicHelper.add_y_and_shift(self.m_h_angle)
        self.graphicHelper.set_text(f"angle: {self.m_h_angle}")
        self.graphicHelper.update()

        # print angle on frame
        cv2.putText(annotated_image, text=str(self.m_h_angle), org=(20, 80), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3,
                    color=(0, 255, 0), thickness=3)

        # send angle to arduino
        self.m_arduinoControl.sendServoAngle(self.m_h_angle, self.m_v_angle)

        return annotated_image

    def __deductAngleFace(self, results):
        """
        Deduct angle for both H and V axis from landmarks. Review https://colab.research.google.com/drive/1uCuA6We9T5r0WljspEHWPHXCT_2bMKUy#scrollTo=BAivyQ_xOtFp
        :param results: results obtained during pose estimation
        :return: estimated H and V angle
        """
        h_angle = self.m_h_angle
        v_angle = self.m_v_angle
        # note that only one face is detected
        if not results.face_landmarks is None and len(results.face_landmarks) > 0:
            # H angle
            # estimate radius: use landmarks for nose and ears
            nose = results.face_landmarks[0][lmn.FACE_NOSE]
            left_ear = results.face_landmarks[0][lmn.FACE_LEFT]
            right_ear = results.face_landmarks[0][lmn.FACE_RIGHT]
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


    
    def loopAimingFace(self, test):
        """
        Runs the main video processing loop for aiming based on face landmarks
        :return:
        """
        logging.info("Entering aiming face base on face landmarks")

        # instantiate arduino controller
        self.m_test = test  
        self.m_arduinoControl = arduinoControl(self.m_test)

        # instantiate face landmarks recognizer
        base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=False,
                                            output_facial_transformation_matrixes=False,
                                            num_faces=1)
        self.m_detector = vision.FaceLandmarker.create_from_options(options)

        # instantiate graphic helper to record angle values
        self.graphicHelper = GraphicsHelper(0, 20, -100, 100)

        try:
            # Create a video looper, that uses the aiming function as the frame processing function
            vl = videoloop(self.__aimingFace)
            # Invoke the main oop
            vl.loopVideoProcess()
        except ValueError:
            logging.error("Error" + str(ValueError))
        logging.info("Exiting aiming face")

    #################################################
    # Aiming section based on body pose             #
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

        # record new angle on graphic helper
        self.graphicHelper.add_y_and_shift(h_angle)
        self.graphicHelper.set_text(f"angle: {h_angle}")
        self.graphicHelper.update()

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

    def loopAiming(self, test):
        """
        Runs the main video processing loop for aiming
        :return:
        """
        logging.info("Entering aiming")

        # instantiate arduino controller
        self.m_test = test  
        self.m_arduinoControl = arduinoControl(self.m_test)

        # instantiate pose recognizer for full boyd pose estimation
        self.initBodyPose()

        # instantiate graphic helper to record angle values
        self.graphicHelper = GraphicsHelper(0, 20, -100, 100)

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

    def __getTrail(self, frame, mask, delay=1):
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
            # generate solid trail for oldest mask. Use plain color
            # remember: trails are colorized images, masks are grey scale
            oldest_solid_trail = np.zeros(frame.shape, dtype=np.uint8)
            oldest_solid_trail[:] = self.MASK_COLOR[self.NUMBER_OF_MASK-1]

            # background: the background is composed by adding the different trails on each iterantion
            # for the first iteration, background is just the oldest trail
            oldest_mask = self.m_prev_mask_array[self.NUMBER_OF_MASK -1]
            background = cv2.bitwise_and(oldest_solid_trail, oldest_solid_trail, mask=oldest_mask)

            # for the newer trails, iterate by composing the trail with the background
            for i in reversed(range(self.NUMBER_OF_MASK - 1)):
                # generate next colorized trail
                current_solid_trail = np.zeros(frame.shape, dtype=np.uint8)
                current_solid_trail[:] = self.MASK_COLOR[i]
                #cv2.imshow('background ' + str(i), background)
                #cv2.imshow('current_solid_trail ' + str(i), current_solid_trail)
                #cv2.imshow('current_mask ' + str(i), self.m_prev_mask_array[i])
                #cv2.waitKey(0)

                # compose the trail with the background using the current mask and set it as current background
                background = self.__composeForeAndBackgroud(current_solid_trail, background, mask= self.m_prev_mask_array[i])

            # update previous mask
            self.m_prev_mask_array[1:self.NUMBER_OF_MASK]= self.m_prev_mask_array[0:self.NUMBER_OF_MASK-1]
            self.m_prev_mask_array[0] = mask
            # save composited trail as previous trail
            self.m_previous_trail = background
            applied = True
        self.m_segmentation_tick += 1
        #cv2.imshow('m_previous_trail ', self.m_previous_trail)
        #cv2.waitKey(0)

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
            #cv2.imshow('mask', mask)

            # show masked image
            # calculate the matching area
            masked = cv2.bitwise_and(frame, frame, mask=mask)
            #cv2.imshow('masked', masked)

            # calculate trailed image
            applied, trail = self.__getTrail(frame, mask)

            # compose frame and trail
            trailed = self.__composeForeAndBackgroud(frame, trail, mask)

            # calculate overall mask
            overall_mask = self.__overallMask(trailed)

            # compose trailed image with background
            final_composition = self.__composeForeAndBackgroud(trailed, frame, overall_mask)
        except ValueError:
            print(str(ValueError))


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

        # instantiate pose recognizer for full boyd pose estimation
        self.initBodyPose()

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
        #self.m_canvas.set_window_title('3D estimation')

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
            mediapipedrawing_utils.plot_landmarks(
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

        # instantiate pose recognizer for full boyd pose estimation
        self.initBodyPose()

        try:
            # Create a video looper, that uses the pose recognition function as the frame processing function
            vl = videoloop(self.__poseRecognition)
            # Invoke the main oop
            vl.loopVideoProcess()
        except ValueError:
            logging.error("Error" + str(ValueError))
        logging.info("Exiting pose recognition")
