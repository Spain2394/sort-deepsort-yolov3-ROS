
## Dependencies
This node uses ```video_stream_opencv```

## Testing
6 FPS worked well

## TODO 
Use small buffers to publish more precise bounding boxes

## parameters

```max_age``` is the maximum number of frames a tracker can exist by making ```max_age > 1``` you can allow it to survive without a detection, for instance if there are n skip frames. set ```max_age = n```. ```min_hits``` is the minimum number of times a tracker must be detected to survive. Default is 3, you can want to reduce to compensate for false detections. 

tracker file format: frame_id,-1,xmin,ymin,w,h,confidence,-1,-1,-1
