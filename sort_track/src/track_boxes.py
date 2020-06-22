#!/usr/bin/env python

"""
ROS node to track objects using SORT TRACKER and YOLOv3 detector (darknet_ros)
Takes detected bounding boxes from darknet_ros and uses them to calculated tracked bounding boxes
Tracked objects and their ID are published to the sort_track node
No delay here
"""

import rospy
import numpy as np
from darknet_ros_msgs.msg import BoundingBoxes
from sort import sort 
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
import time
from sort_track.msg import IntList
import math

def get_parameters():
	"""
	Gets the necessary parameters from .yaml file
	Returns tuple
	"""
	camera_topic = rospy.get_param("~camera_topic")
	detection_topic = rospy.get_param("~detection_topic")
	tracker_topic = rospy.get_param('~tracker_topic')
	cost_threhold = rospy.get_param('~cost_threhold')
	min_hits = rospy.get_param('~min_hits')
	max_age = rospy.get_param('~max_age')
	queue_size = rospy.get_param("~queue_size")
	iou_threshold = rospy.get_param("~iou_threshold")
	display = rospy.get_param("~display")
	fps = rospy.get_param("/video_stream_opencv/fps",30)

	return (queue_size, camera_topic, detection_topic, tracker_topic, cost_threhold, iou_threshold, display, max_age, min_hits, fps)


def callback_det(data):
	global detections
	global trackers
	global track
	global cycle_time
	global start_time
	global total_time
	global detection_event

	detection_event = True
	detections = []
	trackers = []
	track = []

	print("callback det")
	
	for box in data.bounding_boxes:
		detections.append(np.array([box.xmin, box.ymin, box.xmax, box.ymax, round(box.probability,2)]))
		
	detections = np.array(detections)
	#Call the tracker
	trackers = tracker.update(detections,iou_threshold)
	trackers = np.array(trackers, dtype='int')
	track = trackers
	msg.data = track

def callback_image(data):
	
	global display 
	global detections
	global detection_event
	global track
	print("callback image")
	#Display Image
	# if detection_event is False and len(track)!=0: return 
	# else:
	bridge = CvBridge()
	cv_rgb = bridge.imgmsg_to_cv2(data, "bgr8")
	#TO DO: FIND BETTER AND MORE ACCURATE WAY TO SHOW BOUNDING BOXES!!
	#Detection bounding box
	cv2.rectangle(cv_rgb, (int(detections[0][0]), int(detections[0][1])), (int(detections[0][2]), int(detections[0][3])), (100, 255, 50), 1)
	cv2.putText(cv_rgb , "plant", (int(detections[0][0]), int(detections[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 50), lineType=cv2.LINE_AA)
	
	#Tracker bounding box
	cv2.rectangle(cv_rgb, (track[0][0], track[0][1]), (track[0][2], track[0][3]), (255, 255, 255), 1)
	# place id on upp
	cv2.putText(cv_rgb , str(track[0][4]), (track[0][2], track[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
	
	if display:
		cv2.imshow("yolov3-tracker", cv_rgb)
		cv2.waitKey(1)


def main():
	global tracker
	global msg
	global detections
	global total_time
	global total_frames
	global start_time
	global cycle_time
	global tracker_id
	global iou_threshold
	global display
	global detection_event
	display = False
	total_frames = 0
	detection_event = False
	
	# holds tracker id
	counter = IntList()
	while not rospy.is_shutdown():
		#Initialize ROS node
		# frame +=1
		rospy.init_node('sort_tracker', anonymous=False)

		# Get the parameters	
		(queue_size, camera_topic, detection_topic, tracker_topic, cost_threshold, iou_threshold, display, max_age, min_hits, fps) = get_parameters()
		cost_threshold = cost_threshold

		# loop twice per frame
		rate = rospy.Rate(3)


		tracker = sort.Sort(max_age=max_age, min_hits=min_hits) #create instance of the SORT tracker
		
		#Subscribe to darknet_ros to get BoundingBoxes from yolov3-detections
		sub_detection = rospy.Subscriber(detection_topic, BoundingBoxes, callback_det)

		#Subscribe to image topic
		image_sub = rospy.Subscriber(camera_topic,Image,callback_image)

		#Publish results of object tracking
		# pub_trackers = rospy.Publisher(tracker_topic, IntList, queue_size=queue_size)
		# print(msg)
		# 1.5 = 2
		total_frames = math.ceil(total_frames + 1/2)
		#pub_trackers.publish(msg)
		rate.sleep()
		rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
