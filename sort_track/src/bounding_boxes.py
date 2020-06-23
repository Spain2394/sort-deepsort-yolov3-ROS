#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy
import tf

from os.path import expanduser
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBoxes
from sensor_msgs.msg import Image
from std_msgs.msg import Int8

def getParam():
    image_topic = rospy.get_param("~camera_topic")
    detection_topic = rospy.get_param("~detection_topic")
    # object_topic = rospy.get_param("~object_topic")
	# tracker_topic = rospy.get_param('~tracker_topic')
	# cost_threhold = rospy.get_param('~cost_threhold')
	# min_hits = rospy.get_param('~min_hits')
	# max_age = rospy.get_param('~max_age')
	# queue_size = rospy.get_param("~queue_size")
	# iou_threshold = rospy.get_param("~iou_threshold")
	# display = rospy.get_param("~display")
	# fps = rospy.get_param("/video_stream_opencv/fps", 30)

    return image_topic, detection_topic


def image_callback(image):
    """Image callback"""
    # Store value on a private attribute
    global _current_image
    _current_image = image

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
    _detected_bbox = bbox


def run():

    global _current_image
    global _bridge
    global _detected_bbox
    # global _current_objects
    # global coordinates
    global object_class

    _current_image = None
    _detected_bbox = None
    # _current_objects = None
    # create detector
    _bridge = CvBridge()

    # image and point cloud subscribers
    image_topic, detection_topic = getParam()

    # and variables that will hold their values
    rospy.Subscriber(image_topic, Image, image_callback)

    rospy.Subscriber(detection_topic, BoundingBoxes, detector_callback)

    # rospy.Subscriber(object_topic, Int8, object_callback)

    # publisher for frames with detected objects
    _imagepub = rospy.Publisher('~labeled_image', Image, queue_size=10)

    # TODO publish marked image with sort detections


    rospy.loginfo("ready to detect")

    while not rospy.is_shutdown():
            # only run if there is an image
        if _current_image is not None:
            rospy.loginfo("current image received")
            try:
                # convert image from the subscriber into an OpenCV image
                scene = _bridge.imgmsg_to_cv2(_current_image, 'rgb8')
                # publish marked images
                rospy.loginfo("publishing")
                _imagepub.publish(_bridge.cv2_to_imgmsg(scene, 'rgb8'))  # publish detection results

                # what if there is no bounding boxes? 
                if _detected_bbox is not None:
                    for box, in _detected_bbox.bounding_boxes:
                        xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
                        coordinates = np.array([xmin, ymin, xmax, ymax])
                        obj_class = box.Class

                        rospy.loginfo('...' + str(obj_class)+ ' ' + str() + ' at ' + str(coordinates))
                        # 0 for weed 1 for plant 
                        if object_class == 1: 
                            rospy.loginfo("object is a plant, pass bounding box to sort algorithm")
                        # 1. publish image stream to topic labeled_images
                        # 2. draw bounding boxes

            except CvBridgeError as e:
                print(e)


if __name__ == '__main__':
    rospy.init_node('tracking', log_level=rospy.INFO)
    try:
        run()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')