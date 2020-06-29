import numpy as np
import cv2
import time
import sys
import os
import os.path as osp

from utils import visualization_utils as vis_util
from utils import label_map_util
import sort
from PIL import Image
from bounding_boxes import marked_image


def visualization_utils(image, tracker_id, dets):
    global marked_image

    xmin,ymin,xmax,ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    dets = np.array([xmin,ymin,xmax, ymax])

    image_np = np.asarray(image, dtype="int32")

    # is this the right bounding box format ? 
    vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(classes),
            frame_cls_new,
            frame_scores,
            category_index_new, # previously use 'category_index'
            use_normalized_coordinates=True,
            line_thickness=2)

    # print("typ")
        # write to the output video
    return image_np
        # 
    # if is_display is True:
    #     cv2.imshow('video_window',image_np)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

