import pyrealsense2 as rs
import numpy as np
import utils.util_functions as uf

class RealSenseCamera():
	def __init__(self, width, height, fps, device):
		self.res_width = width
		self.res_height = height
		self.fps = fps
		self.pipeline = rs.pipeline()
		self.config = rs.config()
		self.config.enable_device(device)
		self.enable_streams()
		self.align = rs.align(rs.stream.color)
		self.profile = self.pipeline.start(self.config)
		self.dev = self.profile.get_device()
		self.depth_sensor = self.dev.first_depth_sensor()
		self.depth_sensor.set_option(rs.option.visual_preset, 3) # set high accuracy: https://github.com/IntelRealSense/librealsense/issues/2577#issuecomment-432137634
		self.colorizer = rs.colorizer()
		self.colorizer.set_option(rs.option.max_distance,16)#[0-16]
		self.align = rs.align(rs.stream.color)

		######################################################
        #                   filters                          #
        ######################################################
        # https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
        # Filters pipe [Depth Frame >> Decimation Filter >> Depth2Disparity Transform** -> Spatial Filter >> Temporal Filter >> Disparity2Depth Transform** >> Hole Filling Filter >> Filtered Depth. ]
		self.decimation = rs.decimation_filter()
		self.decimation.set_option(rs.option.filter_magnitude, 2)
		self.depth_to_disparity = rs.disparity_transform(True)
		self.spatial = rs.spatial_filter()
		self.spatial.set_option(rs.option.filter_magnitude, 2)
		self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
		self.spatial.set_option(rs.option.filter_smooth_delta, 20)
		self.spatial.set_option(rs.option.holes_fill, 3)
		self.temporal = rs.temporal_filter()
		self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
		self.disparity_to_depth = rs.disparity_transform(False)
		self.hole_filling = rs.hole_filling_filter()

	def enable_streams(self, depth=True, color=True):
		if depth:
			self.config.enable_stream(rs.stream.depth, self.res_width,
										self.res_height, rs.format.z16, self.fps)
		if color:
			self.config.enable_stream(rs.stream.color, self.res_width,
										self.res_height, rs.format.rgb8, self.fps)
		
	def get_images(self, pipeline, filters=True):
		
		depth_frame, color_frame = None, None
		while not depth_frame or not color_frame:
			if(filters):
				for x in range(3):
					#Wait for pair of frames
					frame = pipeline.wait_for_frames()
				
				for x in range(len(frame)):
					frame = self.decimation.process(frame).as_frameset()
					frame = self.depth_to_disparity.process(frame).as_frameset()
					frame = self.spatial.process(frame).as_frameset()
					frame = self.temporal.process(frame).as_frameset()
					frame = self.disparity_to_depth.process(frame).as_frameset()
					frame = self.hole_filling.process(frame).as_frameset()
			else:
			    frame = pipeline.wait_for_frames()
			
			depth_frame = frame.get_depth_frame()
			color_frame = frame.get_color_frame()
		
		# align	
		align = rs.align(rs.stream.color)
		frames = align.process(frame)
		aligned_depth_frame = frames.get_depth_frame()
		
		return color_frame, aligned_depth_frame

	def get_image_data(self, color, depth, colorizer):
		return np.asanyarray(color.get_data()), np.asanyarray(colorizer.colorize(depth).get_data())

	def get_profile(self):
		return self.profile

	def get_intrinsics(self, profile):
		return uf.get_intrinsics(profile)

	def get_colorizer(self):
		return self.colorizer
	
	def get_align(self):
		return self.align

	def get_pipeline(self):
		return self.pipeline
	
	def stop_pipeline(self):
		return self.pipeline.stop()
