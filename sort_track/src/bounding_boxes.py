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
from darknet_ros_msgs.msg import BoundingBoxes, ObjectCount
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
import cv2

from sort import sort
from sort_track.msg import IntList
import time

# def actionclient():
#     client = actionlib.SimpleActionClient('darknet_ros/camera_reading', CheckForObjectsAction, darknet_ros_msgs.msg.CheckForObjectsAction)
#     client.wait_for_server()

def getParam():
    image_topic = rospy.get_param("~camera_topic")
    detection_topic = rospy.get_param("~detection_topic")
    display = rospy.get_param("~display")
    filepath = rospy.get_param("~filepath")
    track_class = rospy.get_param("~track_class")
    # object_topic = rospy.get_param("~object_topic")
	# tracker_topic = rospy.get_param('~tracker_topic')
	# cost_threhold = rospy.get_param('~cost_threhold')
	# min_hits = rospy.get_param('~min_hits')
	# max_age = rospy.get_param('~max_age')
	# queue_size = rospy.get_param("~queue_size")
	# iou_threshold = rospy.get_param("~iou_threshold")
	# display = rospy.get_param("~display")
	# fps = rospy.get_param("/video_stream_opencv/fps", 30)

    return image_topic, detection_topic, display, filepath, track_class


def image_callback(image):
    """Image callback"""
    # Store value on a private attribute
    global _current_image

    _current_image = image
    # _current_image_header_seq = image.header.seq
    # print("image frame: ", _current_image.header.seq)

    

# def object_callback(object):
#     """Image callback"""
#     global _current_objects
#     # Store value on a private attribute
#     _current_objects = object

def detector_callback(bbox):
    
    global _detected_bbox
    
    _detected_bbox = bbox
    
    # print("detection frame: ", _detected_bbox.image_header.seq)

def draw_boxes(bbox, box_name, tracker_id, im):

    # global marked_image
    marked_image = im
    xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    
    cv2.rectangle(marked_image, (xmin,ymin), (xmax, ymax), (0,255,0), 4)
    cv2.putText(marked_image, str(box_name) + " " + str(int(tracker_id)), (xmin - 10, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return marked_image


def flush_cache():
    global _detected_bbox_buffer
    global _current_image_buffer
    global _all_trackers 

    _detected_bbox_buffer[:] = []
    _current_image_buffer[:] = []
    _all_trackers[:] = []

def new_detection(detection):
    global _old_detection
    # print("detection image seq", detection.image_header.seq)
    # print("old detection", _old_detection)
    if detection == None and _old_detection == None: return False
    elif detection is None: return False
    else: return _old_detection != detection.image_header.seq

def cache(detected_bbox, current_image, trackers):
    global _detected_bbox_buffer
    global _current_image_buffer
    global _all_trackers 

    _detected_bbox_buffer.append(detected_bbox)
    _current_image_buffer.append(current_image)
    _all_trackers.append(trackers)

def get_detections(detections):
    dets = []

    if detections is not None:
        for box in detections.bounding_boxes:
            if box.Class == "plant":
                dets.append([box.xmin, box.ymin, box.xmax, box.ymax, box.probability])
        
    if len(dets) == 0: 
        print("tracking an empty detection")

    return dets

def write(all_trackers, current_image_buffer, filepath):
    global _bridge
    global _track_class
    frames = 0
    # 1,2,3,4
    for im in current_image_buffer:
        # 45, 2, 3, for all trackers in a frame
        # marked_image = _bridge.imgmsg_to_cv2(im, 'bgr8')
        for tracker in all_trackers:
            # marked_image = _bridge.imgmsg_to_cv2(im, 'bgr8')
            # print("tracker frame", tracker[-1])
            if int(tracker[-1][0]) == im.header.seq:
                frames +=1
                marked_image = _bridge.imgmsg_to_cv2(im, 'bgr8')
                # marked_image = _bridge.imgmsg_to_cv2(im, 'bgr8')
                for t in tracker[:-1]:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(im.header.seq,t[4],t[0],t[1],t[2],t[3]))
                    bbox = [t[0],t[1],t[2],t[3]]
                    # im should a cv2 image
                    marked_image = draw_boxes(bbox, _track_class, t[4], marked_image)
                cv2.imwrite(filepath + "/" + "im" + str(frames) + ".jpg", marked_image)                
                # return
                # print("frames used", frames)


    # for ic, tr in zip(_cache_image, all_trackers):
    #         marked_image = _bridge.imgmsg_to_cv2(ic, 'bgr8')
    #         for t in tr:
    #             box_t = [t[0], t[1], t[2], t[3]]
    #             draw_detections(box_t, obj_class, t[4], marked_image)
    #         cv2.imwrite(filepath + "/" + "im" + str(frames) + ".jpg", marked_image)
    #         frames +=1
    #         # rospy.loginfo("Saved Image")
    #     # frames = 0


def run():

    # declare global variables that can be updated globally
    global _old_detection
    global _current_image
    global _bridge
    global _detected_bbox
    global _track_class
    # global coordinates
    # global object_class
    # global dets
    global _marked_image
    # global _frame_seq
    # global _current_image_header_seq
    # global _detected_image_header_seq
    # global _tracker
    # global _trackers # contains tracker id, x1, y1, x2, y2, probabilitiy
    global _cache_image
    global _cache_detection
    global _all_trackers
    global _detected_bbox_buffer
    global _current_image_buffer

    # tracker storage
    # _trackers = [] # this is going to hold the state estimates of all of the trackers.
    # _tracker_ids = []
    _all_trackers = []

    # header seq storage
    # _detected_image_header_seq = 0 
    # _current_image_header_seq = -1
    _old_detection = None
    
    # set
    _current_image_buffer = []
    _detected_bbox_buffer = []

    # set cache
    _cache_image = []
    _cache_detection = []

    _marked_image = None

    _current_image = None
    # _detected_bbox_buffer = []
    _detected_bbox = None

    # setup cv2 bridge 
    _bridge = CvBridge()

    # image and point cloud subscribers
    image_topic, detection_topic, display, filepath, _track_class = getParam()

    tracker = sort.Sort(max_age = 10, min_hits=1) #create instance of the SORT tracker
    counter = 0 
    frames = 1 
    duration = 0 
    start_time = 0
        
    boxes = []
    class_ids = []
    detections = 0
    dets = []
    # start_time = time.time()
    # rospy.wait_for_message(detection_topic, BoundingBoxes) 
    # print("ready to detect")
    # # subscribe to raw image feed
    # # subscribe to detections
    rospy.Subscriber(detection_topic, BoundingBoxes, detector_callback, queue_size=1)
    rospy.Subscriber(image_topic, Image, image_callback, queue_size=1)

    tracker = sort.Sort(max_age = 5, min_hits=1) # create instance of the SORT tracker
    _detected_bbox = rospy.wait_for_message(detection_topic, BoundingBoxes)
    # _old_detection = _detected_bbox.image_header.seq

    r = rospy.Rate(6)
    while not rospy.is_shutdown():
        # tracker = sort.Sort(max_age = 10, min_hits=1) #create instance of the SORT tracker
        # old_detection = _detected_bbox.image_header.seq
        # rospy.wait_for_message(detection_topic, BoundingBoxes) 
        # if new_detection(old_detection):
        if new_detection(_detected_bbox) and _current_image is not None:
            dets = get_detections(_detected_bbox)
            # # [box.xmin, box.ymin, box.xmax, box.ymax, box.probability, frame seq]
            trackers = tracker.update(np.array(dets)) # exclude the frame seq
            # # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(_detected_bbox.image_header.seq,t[4],t[0],t[1],t[2],t[3]))
            # # print("len trackers", np.shape(trackers))
            print("detected image", _detected_bbox.image_header.seq)
            print("current image", _current_image.header.seq)
            cache(_detected_bbox, _current_image, np.append(trackers, [[_detected_bbox.image_header.seq]*5],0))
            print("lenghth of current buffer", len(_current_image_buffer))
            if len(_current_image_buffer) > 18:
                write(_all_trackers, _current_image_buffer, filepath) 
                flush_cache()
                return
                # tracker = sort.Sort(max_age = 10, min_hits=1) #create instance of the SORT tracker

                # tracker = sort.Sort(max_age = 10, min_hits=1) #create instance of the SORT tracker
                # return
            _old_detection = _detected_bbox.image_header.seq
        else: pass

        r.sleep()
            # print("waiting for a new detection...")


    # publish image topic with bounding boxes 
    # _imagepub = rospy.Publisher('~labeled_image', Image, queue_size=10)

    # # TODO publish original unmarked frame
    # # _imagepub = rospy.Publisher('~original_frames', Image, queue_size=10)

    # # TODO publish marked image with sort detections

    # rospy.loginfo("ready to detect")

    # in the form: dets.append([x_min, y_min, x_max, y_max, probability])
    # dets = np.empty((0,5))
        # only run if there is an image
    
    # print("detection frame: ", _detected_image_header_seq)
    # print("image frame: ", _current_image_header_seq)

        # if _detected_bbox is not None:
            # print(_detected_bbox)
            # _current_image_buffer.append(_current_image)
            # _detected_bbox_buffer.append(_detected_bbox)
        # write images every 10 seconds
        # print("duration since start: %s" % (time.time() - start_time))
    # if len(_detected_bbox_buffer) == 50:
    #     for i in range(len(_current_image_buffer)):
    #         for d in range(len(_detected_bbox_buffer)):
    #             i_frame_seq = _current_image_buffer[i].header.seq
    #             d_detection_seq = _detected_bbox_buffer[d].image_header.seq
    #             if i_frame_seq == d_detection_seq:
    #                 # print("detection frame: ", i_frame_seq)
    #                 # print("image frame: ", _current_image_buffer[i].header.seq)
    #                 _cache_image.append(_current_image_buffer[i])
    #                 # _cache_detection.append(d_detection_seq)
        
    #     for ic, tr in zip(_cache_image, all_trackers):
    #         marked_image = _bridge.imgmsg_to_cv2(ic, 'bgr8')
    #         for t in tr:
    #             box_t = [t[0], t[1], t[2], t[3]]
    #             draw_detections(box_t, obj_class, t[4], marked_image)
    #         cv2.imwrite(filepath + "/" + "im" + str(frames) + ".jpg", marked_image)
    #         frames +=1
    #         # rospy.loginfo("Saved Image")
    #     # frames = 0
    #     print("plant count:%s"%str(counter))
    #     return
    #     flush() # clear trackers, buffers and cache
    # # what if there is no bounding boxes? 
    # # go through all the detections in each frame
    # if _detected_bbox is not None:
    #     # frames +=1
    #     for box in _detected_bbox.bounding_boxes:
    #         # xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
    #         obj_class = box.Class
    #         # update the trackers one at a time based on detection boxes ? 
    #         # collect all boxes from the relevant class
    #         if box.Class == "plant": 
    #             # rospy.loginfo("cycle_time %s", cycle_time)
    #             # [x1,y1,x2,y2] - for each object
    #             # store the class ids 
    #             dets.append([box.xmin, box.ymin, box.xmax, box.ymax, box.probability])
    #             # class_ids.append(obj_class)
    #             # detections +=
    #         else: pass  # no plants in the image
    # else: print("no detection")

    # dets = np.array(dets) # convert for the n dimensional array
    
    # trackers = tracker.update(dets)
    #     # print(trackers)
    # all_trackers.append(trackers)

    # _detected_bbox_buffer.append(_detected_bbox)
    # _current_image_buffer.append(_current_image)
    
    # iterate through all the trackers
    # for t in trackers:
    #     print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(_detected_image_header_seq,t[4],t[0],t[1],t[2]-t[0],t[3]-t[1]))
    #     # rospy.loginfo("tracker ID: %s", d[4])
    #     # rospy.loginfo("tracker id: " + str(t[4]) + ": " + "info: " + str(t))
    #     # _tracker_ids.append(str(t[4]))
        
    #     str_n = str(int(t[4]))
    #     if (str_n in tracker_ids) == False:
    #         print("unique id found")
    #         # unique id 
    #         tracker_ids.append(str_n)
    #         counter +=1
    #     else: pass

    #     box_t = [t[0], t[1], t[2], t[3]]
    
                    

if __name__ == '__main__':
    rospy.init_node('tracking', log_level=rospy.INFO)
    try:
        run()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')