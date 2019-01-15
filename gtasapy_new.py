import numpy as np
import math
import os
import six.moves.urllib as urllib
import sys
sys.path.insert(0, "E:/workspace_py/object_detection")
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO

from matplotlib import pyplot as plt
from PIL import Image, ImageGrab
import cv2
import time
from directKeys import PressKey, ReleaseKey, W, A, S, D, J
# import pyHook, pythoncom

# Object detection imports
# Here are the imports from the object detection module.
# utils is folder in object_detection/

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017' # model name.
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# Download Model
download_model = False
if download_model:
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
	  file_name = os.path.basename(file.name)
	  if 'frozen_inference_graph.pb' in file_name:
	    tar_file.extract(file, os.getcwd())

# Load a (frozen) Tensorflow model into memory.

# It's unnecessary, but it's a good practice to follow.
# Since a default graph is always registered, every op and variable is placed into the default graph.
# The statement, however, creates a new graph and places everything (declared inside its scope) into this graph.
# If the graph is the only graph, it's useless. But it's a good practice 
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid: 
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print("categories: ", categories)
print("\n\ncategory_index:", category_index)


#########################################################
######################### START #########################
#########################################################

from helper_functions import draw_lines, length_of_line, process_img, roi, straight, left, right, slow_ya_roll
from lane_detection import draw_lanes

# Detection
real_height = 420.0 # experimental value
focal_length = 10.0 # experimental value
brain_ON = True

# Counter of 4 seconds before starting.
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

time_difference = time.time()
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      # ret, image_np = cap.read()

      # FOR GTA-SA
      game_screen = np.asarray(ImageGrab.grab(bbox=(1, 35, 640, 420)))
      # FOR GTAV
      # game_screen = np.asarray(ImageGrab.grab(bbox=(1, 35, 800, 600)))
      game_screen = cv2.cvtColor(game_screen, cv2.COLOR_BGR2RGB)
      image_np = game_screen
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection (inference).
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      # print("num_detections", num_detections)

      # print("boxes : ", boxes)
      # print("\nbox : ", boxes[0][np.argmax(scores)]) # for some reasons boxes is 3d. # batch_size duh.
      # print("\nscore : ", classes[np.argmax(scores)])
      # print("classes : ", classes)

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

      try:
      	vis_util.visualize_boxes_and_labels_on_image_array(
	          image_np,
	          np.squeeze(boxes),
	          np.squeeze(classes).astype(np.int32),
	          np.squeeze(scores),
	          category_index,
	          use_normalized_coordinates=True,
	          line_thickness=2)
      except:
      	pass

      
      image, edgy_image, lines = process_img(image_np)
      if brain_ON:
      	try:
      		# if distance < 20.0: # 40.0 before
	      	# 	ReleaseKey(W)
	      	# 	PressKey(S)
	      	# else:
	      	# 	PressKey(W)
	      	# 	ReleaseKey(S)

	      	l1, l2, m1,m2 = draw_lanes(image_np,lines)
	      	print("slope1 = ", m1)
	      	print("slope2 = ", m2)
	      	straight()
	      	time_difference = time.time() - time_difference
	      	if time_difference >= 1000.0:
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

      cv2.imshow('object detection', image)
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

# cap.close()