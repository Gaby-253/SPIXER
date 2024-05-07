#!/usr/bin/env python

import rospy
import dv_processing as dv
from dvs_msgs.msg import Event, EventArray
from std_msgs.msg import Header

# Open any camera
capture = dv.io.CameraCapture()

# Make sure it supports event stream output, throw an error otherwise
if not capture.isEventStreamAvailable():
    raise RuntimeError("Input camera does not provide an event stream.")

# Initialize a slicer
slicer = dv.EventStreamSlicer()

# Initialize ROS node
rospy.init_node('event_publisher', anonymous=True)

# Initialize ROS publisher
event_publisher = rospy.Publisher('/event_stream', EventArray, queue_size=1)

# Initialize message header
header = Header()

while capture.isRunning():
    events = capture.getNextEventBatch()
    if events is not None:
        event_array_msg = EventArray(header=header, events=[])
        for event in events:
            ros_event = Event(
                x=int(event.x()),
                y=int(event.y()),
                polarity=bool(event.polarity()),
                ts=rospy.Time.from_sec(event.timestamp() / 1e6)  # Convert timestamp to ROS Time
            )
            event_array_msg.events.append(ros_event)
        event_publisher.publish(event_array_msg)