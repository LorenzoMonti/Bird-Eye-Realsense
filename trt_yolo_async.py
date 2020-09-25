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
import queue
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

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

class TrtThread(threading.Thread):
    """TrtThread

    This implements the child thread which continues to read images
    from cam (input) and to do TRT engine inferencing.  The child
    thread stores the input image and detection results into global
    variables and uses a condition varaiable to inform main thread.
    In other words, the TrtThread acts as the producer while the
    main thread is the consumer.
    """
    def __init__(self, distance_queue, frame_queue, cam, conn_device, h, w, model, category_num, master, conf_th):
        """__init__

        # Arguments
            condition: the condition variable used to notify main
                       thread about new frame and detection result
            cam: the camera object for reading input image frames
            model: a string, specifying the TRT SSD model
            conf_th: confidence threshold for detection
        """
        threading.Thread.__init__(self)
        self.cam = cam
        self.conn_device = conn_device
        self.h = h
        self.w = w
        self.model = model
        self.master = master
        self.distance_queue = distance_queue
        self.frame_queue = frame_queue
        self.category_num = category_num
        self.conf_th = conf_th
        self.cuda_ctx = None  # to be created when run
        self.trt_yolo = None   # to be created when run
        self.running = False

    def run(self):
        """Run until 'running' flag is set to False by main thread.

        NOTE: CUDA context is created here, i.e. inside the thread
        which calls CUDA kernels.  In other words, creating CUDA
        context in __init__() doesn't work.
        """

        logging.debug('TrtThread: loading the TRT YOLO engine...')
        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0

        self.trt_yolo = TrtYOLO(self.model, (self.h, self.w), self.category_num)
        colorizer = self.cam.get_colorizer()
        align = self.cam.get_align()
    
        full_scrn = False
        fps = 0.0
        tic = time.time()
        # read balance file
        balance = uf.read_balance_file('./balance_filter.csv')
        intrColor, intrDepth = self.cam.get_intrinsics(self.cam.get_profile())
        profile = self.cam.get_profile()
        
        logging.debug('start running...')
        self.running = True
        while self.running:
            while True:
                color_frame, aligned_depth_frame = self.cam.get_images(self.cam.get_pipeline(), filters=True)                
                #Convert images to numpy arrays
                img, colorized_depth = self.cam.get_image_data(color_frame, aligned_depth_frame, colorizer)
                
                if not self.frame_queue.full():
                    self.frame_queue.put(cv2.resize(img, (600, 338)))

                if self.master:
                    depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)
                else:
                    depth_to_color_extrin = 0
                
                if img is not None:
                    boxes, confs, clss = self.trt_yolo.detect(color_frame, img, self.conf_th)
                    bb_values = calculate_boxes_values(boxes, clss)
                    distances = calculate_distances(aligned_depth_frame, bb_values, profile, balance, intrDepth, self.conn_device, depth_to_color_extrin)
                    social_distance = calculate_social_distances(distances)
                    
                if not self.distance_queue.full():
                    self.distance_queue.put(distances)
                print("current fps: " + str(fps))
                
                toc = time.time()
                curr_fps = 1.0 / (toc - tic)
                # calculate an exponentially decaying average of fps number
                fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
                tic = toc
                

        del self.trt_yolo
        self.cuda_ctx.pop()
        del self.cuda_ctx
        loggin.debug('stopped...')

    def stop(self):
        self.running = False
        self.join()


def birdEyeViewer(distance_queue, distance_queue2, frame_queue, vis_frame, vis):
    logging.debug("Bird-eye viewer")

    dist, dist2, reprj_point = [], [], []

    while True:
        background_eye = np.full((800,525,3), 125, dtype=np.uint8) # background for bird's eye

        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        
        if not distance_queue.empty():
            dist = distance_queue.get()

        if not distance_queue2.empty():
            dist2 = distance_queue2.get()

        """
        # reprojecting 3D points based on master node
        if (len(dist) > 0 and dist[0][-1] != 0): # master node
            depth_to_color_extrin = dist[0][-1] # class 'pyrealsense2.extrinsics'
            #print(depth_to_color_extrin)
            if len(dist2) > 0:
                reprj_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, list(dist2[0][0]))
                #print(reprj_point)
        """

        img = vis.draw_bird_eye(background_eye, dist, dist2, reprj_point)
        cv2.imshow("Bird eye viewer", img)
        
        if vis_frame: # visualize rgb image
            if not frame_queue.empty():
                cv2.imshow("Frame viewer", frame_queue.get())

    return



# calculate x,y, width, height, center_x and center_y, torso_upper_x, torso_upper_y, torso_lower_x, torso_lower_y
def calculate_boxes_values(boxes, clss):
    box_value = []

    for i in range(len(boxes)):
        try:
            if clss[i] == 0: #student class
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

def calculate_distances(aligned_depth_frame, bb_values, profile, balance, intrDepth, conn_device, depth_to_color_extrin):
    distances = []
    for bb in bb_values:
        depth = np.asanyarray(aligned_depth_frame.get_data())
        depth = depth[int(bb[9]):int(bb[7]), int(bb[6]):int(bb[8])].astype(float)
        # Get data scale from the device and convert to meters
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth = depth * depth_scale
        z_axis = np.mean(depth)
        try:
            distances.append([uf.convert_depth_pixel_to_metric_coordinate(uf.depth_optimized(z_axis, balance), float(bb[4]), float(bb[5]), intrDepth), bb[4], bb[5], conn_device, depth_to_color_extrin]) 
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

    #trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)
    
    cuda.init()  # init pycuda driver

    # find connected devices
    connected_devices = uf.find_connected_cameras()
    camera = RealSenseCamera(width, height, fps, connected_devices[0])
    camera.enable_streams(True,True) # depth and color streams

    #camera2 = RealSenseCamera(width, height, fps, connected_devices[1])
    #camera2.enable_streams(True,True) # depth and color streams

    vis = BBoxVisualization(cls_dict)
    
    distance_queue = queue.Queue(100)
    frame_queue = queue.Queue(2)

    distance_queue2 = queue.Queue(100)

    trt_thread = TrtThread(distance_queue, frame_queue, camera, connected_devices[0], h, w, args.model, args.category_num, True, conf_th=0.5).start()
    #time.sleep(.1)
    #trt_thread2 = TrtThread(distance_queue2, camera2, connected_devices[1], h, w, args.model, args.category_num, False, conf_th=0.5).start()

    #frameViewer(frame_queue)
    birdEyeViewer(distance_queue, distance_queue2, frame_queue, True, vis)

    camera.stop_pipeline()
    #camera2.stop_pipeline()  

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
