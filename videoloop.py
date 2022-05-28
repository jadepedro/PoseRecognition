import cv2


class videoloop(object):
    """
    Processes a video source, invoking a frame processing function for each frame
    """

    # debug mode
    m_test = False

    def __init__(self, processingFunction,test=False):
        # Debug mode
        self.m_test=test
        # Camera device
        self.m_cam = cv2.VideoCapture(0)
        # Processing function inside loop
        self.m_processingFunction = processingFunction


    def __getFrame(self, w=640, h=480):
        """
        Retrieves a frame from the configured video device or source
        :param w: width frame output
        :param h: heigh frame output
        :return: frame resized to requested size
        """
        img = None
        try:
            _, myf = self.m_cam.read()
            img = cv2.resize(myf, (w, h))
        except ValueError:
            pass
        return img

    def loopVideoProcess(self):
        """
        Main loop for reading and processing the frame
        :return:
        """

        while self.m_cam.isOpened():
            # Read frame
            frame = self.__getFrame()

            # Process frame through provided function
            if self.m_processingFunction is not None:
                frame = self.m_processingFunction(frame)

            # show the final output
            cv2.imshow('Output', frame)

            if self.commonKeyboardHandle():
                break


    def commonKeyboardHandle(self):
        ret = False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            ret=True
        return ret


