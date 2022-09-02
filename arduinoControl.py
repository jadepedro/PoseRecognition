import logging
import serial
import time

class arduinoControl(object):
    """
    Handles communication with arduino
    """
    m_arduino = None
    m_test = False
    m_last_angle = 0
    ANGLE_THRESHOLD = 15

    def __init__(self, test):
        self.m_test = test
        self.m_last_angle = 0
        logging.info("Test mode for ardunino: " + str(self.m_test))
        if not self.m_test:
            self.m_arduino = serial.Serial(port='COM3', baudrate=115200, timeout=.1)

    def sendServoAngle(self, h_angle, v_angle):
        value = 0
        # round to nearest 10
        h_angle = self.round_to_ten(h_angle)

        # only send if difference above threshold
        if abs(h_angle - self.m_last_angle) >= self.ANGLE_THRESHOLD:
            # update last angle
            self.m_last_angle = h_angle
            if self.m_test:
                logging.info("Angles to be sent to arduino H angle: " + str(h_angle) + " V angle: " + str(v_angle))
            else:
                value = self.write_read(str(90 -h_angle))
        return value

    def write_read(self, x):
        self.m_arduino.write(bytes(x, 'utf-8'))
        data = self.m_arduino.readline()
        return data

    def round_to_ten(self, n):
        rem = n % 10
        if rem < 5:
            n = int(n / 10) * 10
        else:
            n = int((n + 10) / 10) * 10
        return n

