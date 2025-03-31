import os
import numpy as np
import cv2
from subprocess import call

_X_SIZE = 1920 #2304
_Y_SIZE = 1080 #1296

def __draw_limbs(image, keypoints_x, keypoints_y):
	chest = [(1,2), (1,5), (1,8), (2,3), (5,6), (6,7), (3,4)]
	head_neck = [(0,15), (0,16), (0,1)]
	legs_feet = [(8,9), (8,12), (9,10), (12,13), (13,14), (10,11), (11,18), (14,17)]

	h, w = image.shape[:2]
	#15 16 20 21 23 24 

	for joint in head_neck:
		cv2.line(image, (keypoints_x[joint[0]] + int(w / 2), keypoints_y[joint[0]] + int(h / 2)),
			(keypoints_x[joint[1]] + int(w / 2), keypoints_y[joint[1]] + int(h / 2)), color=(10,10,255), thickness=1)

	for joint in chest:
		cv2.line(image, (keypoints_x[joint[0]] + int(w / 2), keypoints_y[joint[0]] + int(h / 2)),
			(keypoints_x[joint[1]] + int(w / 2), keypoints_y[joint[1]] + int(h / 2)), color=(0,255,255), thickness=1)

	for joint in legs_feet:
		cv2.line(image, (keypoints_x[joint[0]] + int(w / 2), keypoints_y[joint[0]] + int(h / 2)),
			(keypoints_x[joint[1]] + int(w / 2), keypoints_y[joint[1]] + int(h / 2)), color=(255,162,80), thickness=1)


def draw(skeleton_keypoints, batch_index, frame_index, path="Batches"):
	keypoints_x = []
	keypoints_y = []

	for i in range(len(skeleton_keypoints)):
		if i % 2 == 0:
			keypoints_x.append(int(skeleton_keypoints[i]*200))
		else:
			keypoints_y.append(int(skeleton_keypoints[i]*300))

	image = np.zeros((_X_SIZE, _Y_SIZE), np.uint8)

	h, w = image.shape[:2]
	image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	
	# Draw centered skeleton keypoints
	for i in range(len(keypoints_x)):
		color = (0,0,0)
		if i == 0 or i == 15 or i == 16:											# Head
			color = (10,10,255)
		elif i == 1 or i == 2 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7:	# Chest
			color = (0,255,255)
		else: 																		# Lower body
			color = (255,162,80)
		cv2.circle(image, (keypoints_x[i] + int(w / 2), keypoints_y[i] + int(h / 2)), radius=3, color=color, thickness=3)

	# Draw limbs between joints
	__draw_limbs(image, keypoints_x, keypoints_y)
	
	# Write frames
	sub_dir = os.path.join(path,str(batch_index))
	if not os.path.exists(sub_dir):
		os.makedirs(sub_dir)
	cv2.imwrite(sub_dir + "/frame_" + str(frame_index) + ".png", image)


def generate_video(batch_index, path):
	 # Generate a video for each batch of frames, requires ffmpeg plugin
	
	print("Writing video for batch: " + str(batch_index))
	call(['ffmpeg', '-framerate', '10', '-i', path + "/" + str(batch_index) + '/frame_%01d.png',
			path + "/" + str(batch_index) + '.avi'])
	print("Video written")