import cv2
import math
import numpy as np
from directKeys import PressKey, ReleaseKey, W, A, S, D, J
from lane_detection import draw_lanes

def draw_lines(img, lines):
	line_lengths = []
	try:
		for line in lines:
			coords = line[0] # cause its stupid
			# print(line)
			# line_lengths.append(length_of_line(coords))
			# if length_of_line(coords) >= 430:
			cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 3)
		# no need to return.
		# print(max(line_lengths))
	except:
		pass
	
	return img

def length_of_line(line):
	x1 = line[0]
	x2 = line[1]
	y1 = line[2]
	y2 = line[3]

	length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

	return length

def process_img(image):
	edgy_image = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), threshold1=50, threshold2=100)

	# BLURRING THE IMAGE TO TO IMPROVE LINE DETECTION (BASICALLY ANTI-ALIASING BEFORE)
	edgy_image = cv2.GaussianBlur(edgy_image, (5, 5), 0)

	# vertices = np.array([[0, 500], [10, 250], [120, 150], [520, 150], [620, 250], [640, 500]])
	# vertices = np.array([[0, 500], [10, 250], [120, 200], [520, 200], [620, 250], [640, 500]])
	vertices = np.array([[0, 500], [10, 250], [120, 230], [520, 230], [620, 250], [640, 500]])
	edgy_image = roi(edgy_image, [vertices])

	# edgy_image = cv2.cvtColor(edgy_image, cv2.COLOR_BGR2GRAY)
	# print(edgy_image[0])

	# edgy_image = edgy_image[:, :, 0] # SLICING TO REMOVE THIRD DIMENSION

    # remember, hough lines algo works better on an edgy line.
    						#							min_line_gap  max_line_gap
	lines = cv2.HoughLinesP(edgy_image, 1, np.pi/180, 180,np.array([]), 100, 5)
	# print(lines.shape)
    # lines contains vertices of each line.
	edgy_image = draw_lines(edgy_image, lines)
	m1 = 0
	m2 = 0
	try:
		l1, l2, m1,m2 = draw_lanes(image,lines)
		cv2.line(image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
		cv2.line(image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
	except Exception as e:
		print(str(e))
		pass

	return image, edgy_image, lines

def roi(img, vertices):
	mask = np.zeros_like(img)
	cv2.fillPoly(mask, vertices, 255)
	masked = cv2.bitwise_and(img, mask)
	return masked

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    # slow_ya_roll()

def left():
    PressKey(A)
    # ReleaseKey(W)
    ReleaseKey(D)
    # ReleaseKey(A)

def right():
    PressKey(D)
    ReleaseKey(A)
    # ReleaseKey(W)
    # ReleaseKey(D)

def slow_ya_roll():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)