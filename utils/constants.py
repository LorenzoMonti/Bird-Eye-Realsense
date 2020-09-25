"""
CONSTANTS
"""
import cv2                                # state of the art computer vision algorithms library
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import numpy as np                        # fundamental package for scientific computing
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API

######################################################
#                  test flag                         #
######################################################
test = False
plot = False
social_distance_viewer = False
personal_distance_viewer = False
bird_eye = False
filter = True
######################################################
#                   colors                           #
######################################################
white = (255, 255, 255)
red = (0, 0, 255)
red_circle = (0, 0, 204)
green = (0, 255, 0)
######################################################
#              setup for darknet model               #
######################################################
weightsPath = "./own_model/yolo-obj_65000.weights" #"./yolov3-tiny.weights"#
configPath = "./own_model/yolo-obj.cfg" #"./yolov3-tiny.cfg"#
width, height, fps = 1280, 720, 30 # optimal resolution

# DARKNET
#net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
######################################################
#               realtime plotting                    #
######################################################
# matplotlib style
plt.style.use('ggplot')
plot_size = 50 # how many points visualize?
x_vec = np.linspace(0, 50, plot_size + 1)[0:-1]
y_vec = []
line1 = []
mean_line = []
######################################################
#                   filters                          #
######################################################
# https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
# Filters pipe [Depth Frame >> Decimation Filter >> Depth2Disparity Transform** -> Spatial Filter >> Temporal Filter >> Disparity2Depth Transform** >> Hole Filling Filter >> Filtered Depth. ]
decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude, 2)
depth_to_disparity = rs.disparity_transform(True)
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
spatial.set_option(rs.option.holes_fill, 3)
temporal = rs.temporal_filter()
temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
disparity_to_depth = rs.disparity_transform(False)
hole_filling = rs.hole_filling_filter()
