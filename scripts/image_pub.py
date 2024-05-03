#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Header
import time

def camera_publisher():
    rospy.init_node('camera_publisher')
    image_pub = rospy.Publisher('camera_image', Image, queue_size=1)
    rate = rospy.Rate(30)  # 10Hz

    cap=cv2.VideoCapture() #'/dev/video0'
    cap.open(0, apiPreference=cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 160)
    cap.set(cv2.CAP_PROP_FPS,30)

    bridge = CvBridge()

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")
            # Add UTC timestamp to the header
            ros_image.header = Header(stamp=rospy.Time.from_sec(time.time()), frame_id='camera')
            image_pub.publish(ros_image)
        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        camera_publisher()
    except rospy.ROSInterruptException:
        pass
