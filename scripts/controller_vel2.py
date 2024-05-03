#!/usr/bin/env python
"""oui
"""
__author__ =  'Gattaux Gabriel <>'

import rospy
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDrive
from board import SCL, SDA
import busio
import time

# Import the PCA9685 module. Available in the bundle and here:
#   https://github.com/adafruit/Adafruit_CircuitPython_PCA9685
from adafruit_motor import servo
from adafruit_motor import motor
from adafruit_pca9685 import PCA9685
# Import the PCA9685 module.
import sys, select, os

VERBOSE=False

class driver:

    def __init__(self):
        '''Initialize ros publisher'''

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/cmd_acker", AckermannDrive, self.callback,  queue_size = 1)
        print("subscribed to /cmd_acker")

        # Set he adress o i2c bus on PCA.
         # Set he adress o i2c bus on PCA.
        i2c = busio.I2C(SCL, SDA)
        pcaserv = PCA9685(i2c,address = 0x40)   #steering I2C address, use i2cdetect to validate this number (0x40)
        pcadc = PCA9685(i2c, address=0x60)      #throttle I2C address, use i2cdetect to validate this number (0x60)
        pcadc.frequency = 200
        pcaserv.frequency = 50

        self.servo7 = servo.Servo(pcaserv.channels[0])

        pcadc.channels[7].duty_cycle = 0xFFFF
        pcadc.channels[0].duty_cycle = 0xFFFF

        self.motor4 = motor.DCMotor(pcadc.channels[5], pcadc.channels[6])
        self.motor4.decay_mode = (
        motor.SLOW_DECAY
        )  # Set motor to active braking mode to improve performance

        self.motor3 = motor.DCMotor(pcadc.channels[2], pcadc.channels[1])
        self.motor3.decay_mode = (
        motor.SLOW_DECAY
        )  # Set motor to active braking mode to improve performance
        
        time.sleep(1)
        self.throttle_d = 0
        self.angular_d = 0


    def callback(self, AckermannDrive):
        '''Callback function of subscribed topic. 
        Here vel get known and write to motors'''
        self.throttle_d = AckermannDrive.speed
        self.angular_d = AckermannDrive.steering_angle

        if self.angular_d > 1:
            self.angular_d = 1
        elif self.angular_d < -1:
            self.angular_d = -1 #=45 1.0472=60
        
        #rospy.loginfo(rospy.get_caller_id() + "I heard %s speed PWM ", str(spe_d_pwm))
        self.servo7.angle = 100 - (self.angular_d*75)
        
        if self.throttle_d > 1:
            self.throttle_d = 1
        elif self.throttle_d < -1:
            self.throttle_d = -1
        self.motor4.throttle = self.throttle_d
        self.motor3.throttle = self.throttle_d

def main(args):
    '''Initializes and cleanup ros node'''
    ic = driver()
    rospy.init_node('AntCar_keyboardteleop')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS ctrl")

if __name__ == '__main__':
    main(sys.argv)
