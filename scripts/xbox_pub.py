#!/usr/bin/env python3
"""oui
"""
__author__ =  'Gattaux Gabriel <>'

import rospy
import pygame
from pygame.locals import *
import time
from ackermann_msgs.msg import AckermannDrive

# Import the PCA9685 module.
import sys, select, os
os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == '__main__':
    rospy.init_node('Antcar_teleop_joystick')

    print("Init...")
    pubcmd = rospy.Publisher('cmd_acker', AckermannDrive, queue_size=1)
    

    print("Init joystick")
    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    for joystick in joysticks:
        print(joystick.get_name())
    time.sleep(1)

    throttle_d = 0
    angle = 0
    phase = 2
    #rate = rospy.Rate(30) # 10hz

    ackermannDrive = AckermannDrive()
    
    while not rospy.is_shutdown():
        for event in pygame.event.get():
            print(event)
            if event.type == JOYBUTTONDOWN:
                if event.button == 1:
                    phase = 2
                    print("Stop the car, searching phase")
                    angle = 0
                    throttle_d = 0
                if event.button == 7:
                    throttle_d = throttle_d + 0.05
                    print(f"Speed {throttle_d}")
                if event.button == 6:
                    throttle_d = throttle_d - 0.05
                    print(f"Speed {throttle_d}")
                if event.button == 0:
                    phase = 1
                    print("Learning phase - Nest Right")
                if event.button == 3:
                    phase = 3
                    print("Exploitation phase")
                if event.button == 8:
                    phase = 6
                    print("Learning Phase - Road")
                if event.button == 4:
                    phase = 5
                    print("Learning Phase - Nest left")
            if event.type == JOYBUTTONUP:
                if event.button == 11:
                    pygame.quit()
                    sys.exit()
                    pca.deinit()
                #print(event)
            if event.type == JOYAXISMOTION:
                #print(event)
                if event.axis == 0:
                    angle = event.value
                    print("Angle : " + str(angle))
                if event.axis == 4:
                    phase = 7
                    print("Learning Base")
                if event.axis == 5:
                    phase = 4
                    print("Learning Feeder")
            # if event.type == JOYHATMOTION:
            #     if event.value == (-1,0):
            #         phase = 7
            #         print("Learning Base")
            #if event.type == JOYHATMOTION:
                #print(event)
            if event.type == JOYDEVICEADDED:
                joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
                for joystick in joysticks:
                    print(joystick.get_name())
            if event.type == JOYDEVICEREMOVED:
                joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
            if event.type == pygame.QUIT: # Note the capitalization
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    pca.deinit()
                    sys.exit()
            if phase != 3:
                ackermannDrive.steering_angle = angle
                ackermannDrive.speed = throttle_d
                pubcmd.publish(ackermannDrive)
            rospy.set_param('phase_nb',phase)
        #rate.sleep()
