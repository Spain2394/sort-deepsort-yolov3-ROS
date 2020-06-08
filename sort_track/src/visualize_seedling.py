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


def visualization_utils(image, tracker_id, dets):
        vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                frame_bbox_array_new,
                frame_cls_new,
                frame_scores,
                category_index_new, # previously use 'category_index'
                use_normalized_coordinates=True,
                line_thickness=2)
            # write to the output video
        vd_writer.write(image_np)
            # 
        if is_display is True:
            cv2.imshow('video_window',image_np)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break