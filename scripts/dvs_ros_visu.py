#!/usr/bin/env python

import rospy
from dvs_msgs.msg import EventArray
import cv2
import numpy as np


def event_callback(event_array):
    # Create an empty image
    image = 255 * np.ones((500, 500, 3), dtype=np.uint8)

    # Draw events on the image
    for event in event_array.events:
        if event.polarity:  # Positive event
            color = (255, 0, 0)  # Blue
        else:  # Negative event
            color = (0, 0, 255)  # Red

        # Draw a circle at the event location
        cv2.circle(image, (event.x, event.y), 1, color, -1)

    # Display the image
    cv2.imshow("Event Stream", image)
    cv2.waitKey(1)  # Wait for a short duration to allow OpenCV to refresh the display

def event_visualizer():
    rospy.init_node('event_visualizer', anonymous=True)

    # Subscribe to the /event_stream topic
    rospy.Subscriber("/event_stream", EventArray, event_callback)

    rospy.spin()

if __name__ == '__main__':
    event_visualizer()
