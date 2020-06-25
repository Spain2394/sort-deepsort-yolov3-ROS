#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import actionlib

# import tf
#include <darknet_ros_msgs/CheckForObjectsAction.h>
# from darknet_ros_msgs.msg import CheckForObjectsAction
from os.path import expanduser
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBoxes
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

    # initilize current image and detected boxes
    _current_image_buffer = []
    _detected_image_buffer = []
    _detected_image_header_seq_buffer = [] 
    _current_image_header_seq_buffer = []

    marked_image = None

    _current_image = None
    # _detected_bbox_buffer = []
    _detected_bbox = None

    # setup cv2 bridge 
    _bridge = CvBridge()

    # image and point cloud subscribers
    image_topic, detection_topic, display = getParam()

    # and variables that will hold their values
    rospy.Subscriber(image_topic, Image, image_callback, queue_size=10)

    rospy.Subscriber(detection_topic, BoundingBoxes, detector_callback, queue_size=10)

    # publisher for frames with detected objects
    # _imagepub = rospy.Publisher('~labeled_image', Image, queue_size=10)

    # TODO publish marked image with sort detections

    rospy.loginfo("ready to detect")
    
    # Loop rate is 10 Hz this should be close to camera parameters
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        # rospy.wait_for_message(detection_topic, BoundingBoxes)
        boxes = []
        class_ids = []
        detections = 0
            # only run if there is an image
        if _current_image is not None:
            rospy.loginfo("current image seq: %s", _current_image_header_seq)
            rospy.loginfo("current detection seq: %s", _detected_image_header_seq)
            if _current_image_header_seq == _detected_image_header_seq:
                print("image seq are the same")
        
            # rospy.loginfo("current image received")
            try:
                # convert image from the subscriber into an OpenCV image
                # initialize the detected image as the current frame
                # _imagepub.publish(_bridge.cv2_to_imgmsg(marked_image, 'rgb8'))  # publish detection results
                marked_image = _bridge.imgmsg_to_cv2(_current_image, 'rgb8')
                # marked_image, objects = self._detector.from_image(scene)  # detect objects
                
                # _imagepub.publish(_bridge.cv2_to_imgmsg(scene, 'rgb8'))  # publish detection results

                # what if there is no bounding boxes? 
                # go through all the detections in each frame
                # if _detected_bbox is not None:

                #     marked_image = _bridge.imgmsg_to_cv2(_current_image, 'rgb8')

                #     for box in _detected_bbox.bounding_boxes:
                #         xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
                #         obj_class = box.Class

                #         # rospy.loginfo(' ' + str(obj_class)+ ' at ' + str(dets))
        
                #         # rospy.loginfo(obj_class == "plant")
                #         if obj_class == "plant": 
                #             boxes.append([xmin, ymin, xmax, ymax])
                #             class_ids.append(obj_class)
                #             detections += 1
                #         else: pass  # no plants in the image
                #         # rospy.loginfo("object is a plant, pass bounding box to sort algorithm")

                #     for i in range(detections):
                #         draw_detections(boxes[i], class_ids[i], marked_image)
                # # publish after each image
                #     _imagepub.publish(_bridge.cv2_to_imgmsg(marked_image, 'rgb8'))  # publish detection results                    
            except CvBridgeError as e:
                print(e)

            # r.sleep()


if __name__ == '__main__':
    rospy.init_node('tracking', log_level=rospy.INFO)
    try:
        run()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')