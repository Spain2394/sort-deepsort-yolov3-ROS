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
import time
# from visualize_seedling import * 

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

def detected_image_callback(image):
    """Image callback"""
    # Store value on a private attribute
    global _current_detected_image
    global _current_detected_image_seq
    # global _current_image_header_seq

    _current_detected_image =  image
    _current_detected_image_seq = image.header.seq
    

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

def draw_detections(box, obj_class, tracker_id, im):

    # draw according to the class id
    # colours = np.random.rand(32, 2) 
    # color = list(np.random.random(size=3) * 256)
    
    # print("Drawing boxes...")
    # print(box)
    # rospy.loginfo("tracker ID: %s", tracker_id)

    global marked_image
    marked_image = im

    thickness = 4
    font_scale = 1
    box_color = (255,0,0)
    text_color = (0,0,0)
    global counter

    xmin,ymin,xmax,ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    
    cv2.rectangle(marked_image, (xmin,ymin), (xmax, ymax), color=box_color, thickness=thickness)
    # txt = f"{obj_class}: {tracker_id:.2f}"
    txt = str(obj_class) + " : " + str(tracker_id)
    (text_width, text_height) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]

    text_offset_x = xmin
    text_offset_y = ymin - 5
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width, text_offset_y - text_height))
    overlay = marked_image.copy()
    # x_center = xmin,  
    
    cv2.rectangle(overlay, box_coords[0], box_coords[1], color=box_color, thickness=cv2.FILLED)

    marked_image = cv2.addWeighted(overlay, 0.6, marked_image, 0.4, 0)
    cv2.putText(marked_image, txt, (xmin - 5, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=text_color, thickness=thickness)
    # cv2.putText(marked_image, counter, (xmax - 5, ymax - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(255,255,255), thickness=thickness)
    # cv2.putText(marked_image, txt, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=text_color, thickness=thickness)



def new_detection(detection):
    global _old_detection
    # print("detection image seq", detection.image_header.seq)
    # print("old detection", _old_detection)
    if detection == None and _old_detection == None: return False
    elif detection is None: return False
    else: return _old_detection != detection.image_header.seq

def run():

    # declare global variables that can be updated globally
    global counter 
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
    global _current_detected_image
    global tracker
    global trackers # contains tracker id, x1, y1, x2, y2, probabilitiy
    global _old_detection
    global _current_detected_image_seq
    trackers = [] # this is going to hold the state estimates of all of the trackers.
    tracker_ids = []

    _old_detection = None
    _current_detected_image_seq = -1
    _detected_image_header_seq = 0 
    _current_image_header_seq = -1
    temp_detected_image = -2
    counter = 0

    # initilize current image and detected boxes
    _current_image_buffer = []
    _detected_image_buffer = []
    _detected_image_header_seq_buffer = [] 
    _current_image_header_seq_buffer = []

    marked_image = None

    _current_image = None
    _current_detected_image = None
    # _detected_bbox_buffer = []
    _detected_bbox = None

    # setup cv2 bridge 
    _bridge = CvBridge()

    # image and point cloud subscribers
    image_topic, detection_topic, display = getParam()


    # subscribe to raw image feed
    # rospy.Subscriber(image_topic, Image, image_callback, queue_size=100)

    # subscribe to detections
    rospy.Subscriber(detection_topic, BoundingBoxes, detector_callback, queue_size=1)

    rospy.Subscriber("/darknet_ros/detection_image", Image, detected_image_callback, queue_size=1)

    

    # publish image topic with bounding boxes 
    _imagepub = rospy.Publisher('~labeled_image', Image, queue_size=1)

    # TODO publish original unmarked frame
    # _imagepub = rospy.Publisher('~original_frames', Image, queue_size=10)

    # TODO publish marked image with sort detections

    rospy.loginfo("ready to detect")

    # adjust frame rate 

    

    r = rospy.Rate(6)
    # max age is the maximum number of frames a tracker can exist by making max_age > 1 
    # you can allow it to survive without a detection, for instance if there are skip frames 
    # in this there is an expected number of skip frames, by making max_age = n, you are allowing
    # for n skip frames. 
    # min hits is the minimum number of times a tracker must be detected to survive
    
    tracker = sort.Sort(max_age = 8, min_hits=1) #create instance of the SORT tracker
    # counter = 0 
    frames = 1 

    _old_detection = rospy.wait_for_message(detection_topic, BoundingBoxes).image_header.seq
    # rospy.wait_for_message("/darknet_ros/detection_image", Image)
    while not rospy.is_shutdown():
        # _old_detection = _detected_bbox.image_header.seq
        # rospy.wait_for_message("/darknet_ros/detection_image", Image)
        
        boxes = []
        class_ids = []
        detections = 0
        dets = []
        # if frames == 1: 
        #     _old_detection = _detected_bbox.seq

        # if frames == 1: 
        #     _old_detection = _detected_bbox

        # in the form: dets.append([x_min, y_min, x_max, y_max, probability])
        # dets = np.empty((0,5))
            # only run if there is an image
        if  new_detection(_detected_bbox) and _current_detected_image is not None:

            # if frames == 1: 
            #     _old_detection = _detected_bbox.image_header.seq
            
            print("current image seq: %s", _current_detected_image_seq)
            # print("frames processed %s: " %frames)
            print("detected frame number: %s" % _detected_image_header_seq)
            # if frame < 3 : 
            #     rospy.spin()
            
            # check to see if curr detection is on the current frame
            # if _current_image_header_seq == _detected_image_header_seq:
                # print("image seq are the same")
            
            try:
                # image received
                # convert image from the subscriber into an OpenCV image
                # initialize the detected image as the current frame
                # initialize our current frame 
                # marked_image = _bridge.imgmsg_to_cv2(_current_image, 
                # 'rgb8')

                marked_image = _bridge.imgmsg_to_cv2(_current_detected_image, 
                'rgb8')
                # _imagepub.publish(_bridge.cv2_to_imgmsg(marked_image, 'rgb8'))  # publish detection results
                
                # marked_image, objects = self._detector.from_image(scene)  # detect objects
                
                # _imagepub.publish(_bridge.cv2_to_imgmsg(scene, 'rgb8'))  # publish detection results

                # what if there is no bounding boxes? 
                # go through all the detections in each frame
                if _detected_bbox is not None:
                    frames +=1
                    for box in _detected_bbox.bounding_boxes:
                        # xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
                        obj_class = box.Class
                        # update the trackers one at a time based on detection boxes ? 
                        # collect all boxes from the relevant class
                        if box.Class == "plant": 
                            # rospy.loginfo("cycle_time %s", cycle_time)
                            # [x1,y1,x2,y2] - for each object
                            # store the class ids 
                            dets.append([box.xmin, box.ymin, box.xmax, box.ymax, box.probability])
                            # class_ids.append(obj_class)
                            # detections +=
                        else: pass  # no plants in the image

                    
                    # there are no detections, you still need to update you trackers
                dets = np.array(dets) # convert for the n dimensional array
                # whether or not the detections are none, the trackers still need to be updated
                # if len(dets) == 0:
                #     rospy.loginfo("no targets detected!")
                #     # trackers = tracker.update(dets)
                # else: 
                #     rospy.loginfo("targets detected")
                    # trackers = tracker.update(dets)
                # rospy.loginfo("trackers updated based on current detections")
                trackers = tracker.update(dets)
                # print(trackers)
            
                # iterate through all the trackers
                for t in trackers:
                    # rospy.loginfo("tracker ID: %s", d[4])
                    # rospy.loginfo("tracker id: " + str(t[4]) + ": " + "info: " + str(t))
                    # _tracker_ids.append(str(t[4]))
                    str_n = str(int(t[4]))
                    if (str_n in tracker_ids) == False:
                        # print("unique id found")
                        # unique id 
                        tracker_ids.append(str_n)
                        counter +=1
                    else: pass

                    box_t = [t[0], t[1], t[2], t[3]]
                    # draw every tracker 
                    # draw_detections(box, obj_class, tracker_id, im):
                    # draw_detections(box_t, obj_class, t[4], marked_image)
                    # draw_detections(marked_image, t[4], box_t)
                    draw_detections(box_t, "plant", t[4], marked_image)
                
                print("plant count:%s"%str(counter))
                # the default case is the current frame with no writting
                _imagepub.publish(_bridge.cv2_to_imgmsg(marked_image, 'rgb8'))  # publish detection results                    
                _old_detection = _detected_bbox.image_header.seq

            except CvBridgeError as e:
                print(e)

            r.sleep() # best effort to maintain loop rate r for each frame


if __name__ == '__main__':
    rospy.init_node('tracking', log_level=rospy.INFO)
    try:
        run()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')