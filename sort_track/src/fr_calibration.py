#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import actionlib
import time 

# import tf
#include <darknet_ros_msgs/CheckForObjectsAction.h>
# from darknet_ros_msgs.msg import CheckForObjectsAction
from os.path import expanduser
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBoxes, ObjectCount
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
import cv2

from sort import sort
from sort_track.msg import IntList

# def actionclient():
#     client = actionlib.SimpleActionClient('darknet_ros/camera_reading', CheckForObjectsAction, darknet_ros_msgs.msg.CheckForObjectsAction)
#     client.wait_for_server()

def getParam():
    image_topic = rospy.get_param("~camera_topic")
    detection_topic = rospy.get_param("~detection_topic")
    display = rospy.get_param("~display")
    # object_topic = rospy.get_param("~object_topic")
	# tracker_topic = rospy.get_param('~tracker_topic')
	# cost_threhold = rospy.get_param('~cost_threhold')
	# min_hits = rospy.get_param('~min_hits')
	# max_age = rospy.get_param('~max_age')
	# queue_size = rospy.get_param("~queue_size")
	# iou_threshold = rospy.get_param("~iou_threshold")
	# display = rospy.get_param("~display")
	# fps = rospy.get_param("/video_stream_opencv/fps", 30)

    return image_topic, detection_topic, display


def image_callback(image):
    """Image callback"""
    # Store value on a private attribute
    print("image callback")
    global _current_image
    global _current_image_header_seq

    _current_image = image
    _current_image_header_seq = image.header.seq
    

# def object_callback(object):
#     """Image callback"""
#     global _current_objects
#     # Store value on a private attribute
#     _current_objects = object

def detector_callback(bbox):
    """Point cloud callback"""
    # Store value on a private attribute
    # in form 
    global _detected_bbox
    global _detected_image_header_seq

    _detected_bbox = bbox
    _detected_image_header_seq = bbox.image_header.seq

def draw_detections(box, class_id, im):
    global marked_image
    marked_image = im

    cv2.rectangle(marked_image,(box[0],box[1]),(box[2], box[3]), (0,255,0),5)
    cv2.putText(marked_image, class_id, (box[0] - 10, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)


def run():

    # declare global variables that can be updated globally
    global _current_image
    global _bridge
    global _detected_bbox
    global coordinates
    global object_class
    global dets
    global marked_image
    global _frame_seq
    global _current_image_header_seq
    global _detected_image_header_seq

    _detected_image_header_seq = 0 
    _current_image_header_seq = -1
    temp_detected_image = -2

    marked_image = None

    _current_image = None
    # _detected_bbox_buffer = []
    _detected_bbox = None

    # setup cv2 bridge 
    _bridge = CvBridge()

    image_topic, detection_topic, display = getParam()

    # rospy.wait_for_message("/darknet_ros/detection_image", Image)

    # and variables that will hold their values
    # rospy.Subscriber(image_topic, Image, image_callback, queue_size=10)

    rospy.Subscriber(detection_topic, BoundingBoxes, detector_callback, queue_size=10)

    FPS = 0
    calibration = False
    start_time = time.time()
    frames = 0
    while not rospy.is_shutdown() and calibration is False:
        rospy.wait_for_message(detection_topic, BoundingBoxes)
        frames +=1
        if frames == 500: 
            calibration = True
        # rospy.spin()

    print("FPS: ", 500 / (time.time() - start_time))

if __name__ == '__main__':
    rospy.init_node('tracking', log_level=rospy.INFO)
    try:
        run()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')