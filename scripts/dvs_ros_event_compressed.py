#!/usr/bin/env python

import rospy
import dv_processing as dv
from dvs_msgs.msg import Event, EventArray
from std_msgs.msg import Header
import pickle
import zlib

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
    list_of_events = []

    if events is not None:
        for event in events:
            list_of_events.append([int(event.x()), int(event.y()), bool(event.polarity()), rospy.Time.from_sec(event.timestamp() / 1e6)])
        dumped_events = pickle.dumps(list_of_events)
        compressed_events = zlib.compress(dumped_events)
        event_publisher.publish(compressed_events)


# # receive Events
# list_of_events = gzip.decompress(received_data)
# event_array_msg = EventArray(header=list_of_events.pop(0), events=[Event(*event_i) for event_i in list_of_events])
