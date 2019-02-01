import numpy as np

from matplotlib import pyplot as plt
from PIL import Image, ImageGrab
import cv2
import time
from directKeys import PressKey, ReleaseKey, W, A, S, D, J
# import pyHook, pythoncom

#########################################################
######################### START #########################
#########################################################

from helper_functions import draw_lines, length_of_line, process_img, roi, straight, left, right, slow_ya_roll
from lane_detection import draw_lanes
from coco_object_detection import ObjectDetection

real_height = 420.0 # experimental value
focal_length = 10.0 # experimental value
brain_ON = True

objectDetection = ObjectDetection()
objectDetection.initialize()


# Counter of 4 seconds before starting.
for i in list(range(4))[::-1]:
	print(i+1)
	time.sleep(1)

time_difference = time.time()

while True:
	# ret, image_np = cap.read()

	# FOR GTA-SA
	game_screen = np.asarray(ImageGrab.grab(bbox=(1, 35, 640, 420)))
	# FOR GTAV
	# game_screen = np.asarray(ImageGrab.grab(bbox=(1, 35, 800, 600)))
	game_screen = cv2.cvtColor(game_screen, cv2.COLOR_BGR2RGB)
	# image_np = game_screen
	
	boxes, scores, classes, num_detections, image_with_detections = objectDetection.do_inference(game_screen, return_visualized_image=True)   

	box_with_max_score = boxes[0][np.argmax(scores)] # [ymin, xmin, ymax, xmax]
	apparent_height = (box_with_max_score[2] - box_with_max_score[0])*420.0 # multiplying to de-normalize boxes.
	print("\nApparent height = ", apparent_height)
	print("\nApparent height (normalized) = ", apparent_height / 420.0)
	distance = ((real_height*focal_length)/apparent_height) - focal_length
	# velocity = (distance - )/5
	print("\nDistance from object = ", distance)


	# print(f"Detection Logs [boxes, classes, scores]: {boxes, classes, scores}")

	# cv2.imshow('object detection', cv2.resize(image_np, (640,420)))

	# image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

	# image_np = process_img(image_np)
	# image_np = cv2.resize(image_np, (640, 420))

	processed_frame, edgy_image, lines = process_img(game_screen)
	if brain_ON:
		try:
			# if distance < 20.0: # 40.0 before
			# 	ReleaseKey(W)
			# 	PressKey(S)
			# else:
			# 	PressKey(W)
			# 	ReleaseKey(S)

			l1, l2, m1,m2 = draw_lanes(processed_frame,lines)
			print("slope1 = ", m1)
			print("slope2 = ", m2)
			straight()
			time_difference = time.time() - time_difference
			if time_difference >= 3000.0:
				slow_ya_roll()
				time.sleep(0.5)
			
			if m1*m2 >= 0.0: # same side
				if m1 < 0.0:
					right()
			# 		ReleaseKey(A)
			# 		PressKey(W)
			# 		PressKey(D)
			# 		ReleaseKey(W)
				else:
					left()
			# 		ReleaseKey(D)
			# 		PressKey(W)
			# 		PressKey(A)
			# 		ReleaseKey(W)


		except Exception as e:
			print("error in brain : ", str(e))
			pass

		cv2.imshow('object detection', image_with_detections)
		# FOR GTA-SA
		cv2.imshow('process_img', cv2.resize(edgy_image, (640,420)))
		# FOR GTA V
		# cv2.imshow('process_img', cv2.resize(edgy_image, (800,600)))
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
		elif cv2.waitKey(25) & 0xFF == ord('b'):
			brain_ON = not brain_ON
			print("b pressed.")