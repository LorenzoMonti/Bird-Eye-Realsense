import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import math
import time

# Constants
COLS = 1280
ROWS = 720

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
								PLOT UTIL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def plot_data(z_axis, x_vec, y_vec, plot_size, line1, mean_line, title):
	if(len(y_vec) >= plot_size):
		y_vec[-1] = z_axis
		line1, mean_line = live_plotter(x_vec, y_vec, line1, mean_line, title)
		y_vec = np.append(y_vec[1:],0.0)
	else:
		y_vec.append(z_axis)

def live_plotter(x_vec, y1_data, line1, mean_line, identifier='', pause_time=0.001):
	if line1==[]:
		# this is the call to matplotlib that allows dynamic plotting
		plt.ion()
		fig = plt.figure(figsize=(13,6))
		ax = fig.add_subplot(111)
		# create a variable for the line so we can later update it
		line1, = ax.plot(x_vec, y1_data, '-o', alpha=0.8)
		mean_line, = ax.plot(x_vec, [np.mean(y1_data)] * len(x_vec), label='Mean', linestyle='--')
		ax.legend((line1, line1), ('mean:' + str(np.mean(y1_data)), 'std:' + str(np.std(y1_data))))
		#update plot label/title
		plt.ylabel('Z axis')
		plt.title('{}'.format(identifier))
		plt.show()

	# after the figure, axis, and line are created, we only need to update the y-data
	line1.set_ydata(y1_data)
	mean_line.set_ydata([np.mean(y1_data)] * len(x_vec))
	plt.legend((line1, line1), ('mean:' + str(np.mean(y1_data)), 'std:' + str(np.std(y1_data))))

	# adjust limits if new data goes beyond bounds
	if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
		plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])

	# this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
	plt.pause(pause_time)

	# return line so we can update it again in the next iteration
	return line1, mean_line


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
								UTILITIES
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def get_intrinsics(profile):
	# Get intrinsics
	depth_sensor = profile.get_device().first_depth_sensor()
	cam_scale = depth_sensor.get_depth_scale()
	intrc =  profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
	intrd =  profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
	return intrc, intrd
	
# find connected cameras
def find_connected_cameras():
	realsense_ctx = rs.context()
	connected_devices = []
	for i in range(len(realsense_ctx.devices)):
		detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
		connected_devices.append(detected_camera)
	return connected_devices

# get_extrinsics_to(self: pyrealsense2.stream_profile, to: pyrealsense2.stream_profile) → pyrealsense2.extrinsics,
#Get the extrinsic transformation between two profiles (representing physical sensors)

# get_device_extrinsics(from_stream, to_stream)
# Retrieve extrinsic transformation between the viewpoints of two different streams.

# rs2_transform_point_to_point(...)
# Transform 3D coordinates relative to one sensor to 3D coordinates relative to another viewpoint

def read_balance_file(file):
	balance = []
	with open(file, 'r') as csvfile:
		for line in csvfile.readlines():
			col = line.split(',')
			balance.append([float(col[0]),float(col[1])])
	return balance


def depth_optimized(depth, balance):
	index = 0
	for i in balance:
		if (depth >= i[0]):
			index = i
	depth = depth + index[1]
	return depth

# https://www.calvert.ch/maurice/improving-the-depth-map-accuracy-of-realsense-cameras-by-an-order-of-magnitude/
# https://www.calvert.ch/maurice/2018/11/01/realsense-cameras-calculating-3d-coordinates-from-depth-row-and-column/
def convert_row_col_range_to_point(depth, row, col):
	vratio = (VCENTRE - row) * VPIXEL
	hratio = (col - HCENTRE) * HPIXEL
	z = depth + A * depth * depth + B * depth #+ C
	y = depth * (-vratio)
	x = depth * hratio
	return x, y, z


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
				Euclidian distance method (3D)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
Convert the depth and image point information to metric coordinates
Parameters:
-----------
depth 	 	 	 : double
					   The depth value of the image point
pixel_x 	  	 	 : double
					   The x value of the image coordinate
pixel_y 	  	 	 : double
						The y value of the image coordinate
camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
Return:
----------
X : double
	The x value in meters
Y : double
	The y value in meters
Z : double
	The z value in meters
"""
def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y, camera_intrinsics):
	X = (pixel_x - camera_intrinsics.ppx)/camera_intrinsics.fx * depth
	Y = (pixel_y - camera_intrinsics.ppy)/camera_intrinsics.fy * depth
	return X, Y, depth


"""
Get Euclidian distance between two 3D points
-----------
point1 	 	 	 	: list(tuple)
point2 	  	 	 : list(tuple)

Return:
----------
euclidian distance : double

"""
def euclidian_distance(point1, point2):
	point1_X, point1_Y, point1_Z = point1[0]
	point2_X, point2_Y, point2_Z = point2[0]
	return(math.sqrt( (point2_X - point1_X)**2 + (point2_Y - point1_Y)**2 + (point2_Z - point1_Z)**2 ))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
						Carnot Method (2D)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# https://medium.com/@manivannan_data/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
def getAngle(left, center, right):
	a = np.array(left)
	b = np.array(center)
	c = np.array(right)

	ba = a - b
	bc = c - b

	cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
	angle = np.arccos(cosine_angle)

	return np.degrees(angle)

def carnot(first, second, angle):
	return math.sqrt(((np.square(first) + np.square(second)) - (2 * first * second * math.cos(angle))))
