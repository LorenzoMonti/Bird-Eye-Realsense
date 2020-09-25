"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse
import threading

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import pycuda.driver as cuda                                                
from itertools import combinations

from utils.yolo_classes import get_cls_dict
from utils.yolo import TrtYOLO
from utils.realsensecamera import *
#from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
import utils.util_functions as uf
from utils.constants import *
import pyrealsense2 as rs

WINDOW_NAME = 'TrtYOLODemo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    #parser = add_camera_args(parser)
    parser.add_argument(
        '--model', type=str, default='yolov3-spp-608', required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp-608|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    parser.add_argument(
        '--category_num', type=int, default=80,
        help='number of object categories [80]')
    args = parser.parse_args()
    return args


def loop_and_detect(camera, pipeline, profile, trt_yolo, WINDOW_NAME, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """

    colorizer = camera.get_colorizer()
    align = camera.get_align()
    intrColor, intrDepth = camera.get_intrinsics(profile)
    
    # read balance file
    balance = uf.read_balance_file('./balance_filter.csv')

    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:    

        #if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
        #    break
        
        color_frame, aligned_depth_frame = camera.get_images(pipeline, filters=False)
        #Convert images to numpy arrays
        img, colorized_depth = camera.get_image_data(color_frame, aligned_depth_frame, colorizer)
        key = cv2.waitKey(1)
        print(color_frame)
        if img is not None:
            boxes, confs, clss = trt_yolo.detect(color_frame, img, conf_th)
            bb_values = calculate_boxes_values(boxes)
            img = vis.draw_bboxes(colorized_depth, boxes, confs, clss)
            img = vis.draw_centers(colorized_depth, bb_values)
            img = vis.draw_torso(colorized_depth, bb_values)
            distances = calculate_distances(aligned_depth_frame, bb_values, profile, balance, intrDepth)
            social_distance = calculate_social_distances(distances)
            img = vis.draw_distances(colorized_depth, distances)
            img = vis.draw_social_distance(colorized_depth, social_distance, 0.1)
            img = show_fps(colorized_depth, fps)
            cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
        
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

# calculate x,y, width, height, center_x and center_y, torso_upper_x, torso_upper_y, torso_lower_x, torso_lower_y
def calculate_boxes_values(boxes):
    box_value = []

    for i in range(len(boxes)):
        try:
            width = int((boxes[i,2] - boxes[i,0]))
            height = int((boxes[i,3] - boxes[i,1]))
            center_x = (boxes[i,0] + width / 2)
            center_y = (boxes[i,1] + height / 2)
            x = int(center_x - (width / 2))
            y = int(center_y - (height / 2))
            (torso_upper_x, torso_upper_y) = (center_x - (height/12), center_y + (height/8))
            (torso_lower_x, torso_lower_y) = (center_x + (height/12), center_y - (height/8))
            box_value.append([x, y, int(width), int(height), center_x, center_y, torso_upper_x, torso_upper_y, torso_lower_x, torso_lower_y])
        except IndexError:
            pass
    return box_value 

def calculate_distances(aligned_depth_frame, bb_values, profile, balance, intrDepth):
    distances = []
    for bb in bb_values:
        depth = np.asanyarray(aligned_depth_frame.get_data())
        depth = depth[int(bb[9]):int(bb[7]), int(bb[6]):int(bb[8])].astype(float)
        # Get data scale from the device and convert to meters
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth = depth * depth_scale
        z_axis = np.mean(depth)
        try:
            distances.append([uf.convert_depth_pixel_to_metric_coordinate(uf.depth_optimized(z_axis, balance), float(bb[4]), float(bb[5]), intrDepth), bb[4], bb[5]]) 
        except RuntimeError:
            pass
    
    return distances

def calculate_social_distances(distances):
    social_distance = []
    if (len(distances) >= 2):
        # combinations of every bboxes found. Usare il coefficiente binomiale per calcolare le combinazioni rispetto ai bboxes trovati (54 bboxes = 1431)
        comb = combinations(distances, 2)
        for i in list(comb):
            social_distance.append([uf.euclidian_distance(i[0][:1], i[1][:1]), i[0][1], i[0][2], i[1][1], i[1][2]])
    return social_distance

def thread_camera(connected_device, trt_yolo, conf_th, cls_dict):
    cuda_ctx = cuda.Device(0).make_context()  # GPU 0
     # setup
    camera = RealSenseCamera(width, height, fps, connected_device)
    camera.enable_streams(True,True) # depth and color streams
    
    vis = BBoxVisualization(cls_dict)
    open_window(WINDOW_NAME, 640, 480,'TensorRT Social Distancing ')
    
    loop_and_detect(camera, camera.get_pipeline(), camera.get_profile(), trt_yolo, WINDOW_NAME, conf_th=conf_th, vis=vis)
    del trt_yolo
    cuda_ctx.pop()
    del cuda_ctx
    camera.stop_pipeline()


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    # classes and model
    cls_dict = get_cls_dict(args.category_num)
    yolo_dim = args.model.split('-')[-1]

    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)
    
    cuda.init()  # init pycuda driver
         
    #vis = BBoxVisualization(cls_dict)

    # find connected devices
    connected_devices = uf.find_connected_cameras()
    print(connected_devices)
    thread_camera(connected_devices[0], trt_yolo, 0.5, cls_dict)
    #thread_camera(connected_devices[1], trt_yolo, 0.5, cls_dict)

    #open_win = threading.Thread(target=open_window, args=[WINDOW_NAME + connected_devices, 640, 480,'TensorRT Social Distancing ' + connected_devices])
    #threading.Thread(target=thread_camera, args=[connected_devices[0], trt_yolo, 0.5, cls_dict]).start()
    #time.sleep(0.5)
    #threading.Thread(target=thread_camera, args=[connected_devices[1], trt_yolo, 0.5, cls_dict]).start()                                                      
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
