import logging


class arduinoControl(object):
    """
    Hanldes communication with arduino
    """

    m_test = False
    def __init__(self, test):
        self.m_test = test

    def sendServoAngle(self, h_angle, v_angle):
        if self.m_test:
            logging.debug("Angles to be sent to arduino H angle: " + str(h_angle) + " V angle: " + str(v_angle))
