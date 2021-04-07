import pyrealsense2 as rs
import numpy as np
import cv2
from ar_markers import detect_markers

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        ########## detect ar markers
        markers = detect_markers(color_image)
        for marker in markers:
            marker.highlite_marker(color_image)
        cv2.imshow('test frame',color_image)

        # wc = depth_image.shape[1]//2
        # hc = depth_image.shape[0]//2
        # center_distance = aligned_depth_frame.get_distance(wc,hc)
        # print('Center distance : {:.3f}m'.format(center_distance))

        # ######## Find red rectangle in the image
        # hsvimg = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        # lower_range = np.array([175,50,20])
        # upper_range = np.array([180,255,255])
        # lower_range2 = np.array([0,50,20])
        # upper_range2 = np.array([5,255,255])

        # mask = cv2.inRange(hsvimg, lower_range, upper_range)
        # mask2 = cv2.inRange(hsvimg, lower_range2, upper_range2)
        # sum_mask = cv2.bitwise_or(mask, mask2)
        # masked_img = cv2.bitwise_and(color_image, color_image ,mask=sum_mask)
        # masked_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        # ######## Contour
        # contours, hierarchy = cv2.findContours(masked_img_gray, cv2.RETR_EXTERNAL,
        #                                         cv2.CHAIN_APPROX_SIMPLE)
        # for contour in contours:
        #     masked_img = cv2.drawContours(masked_img, [contour], -1, (0,0,255), 2)

        # cv2.imshow("image",masked_img)

        # Remove background - Set pixels further than clipping_distance to grey
        # grey_color = 153
        # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # images = np.hstack((bg_removed, depth_colormap))

        # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        # cv2.imshow('Align Example', images)
        
        # Press esc or 'q' to close the image window
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()