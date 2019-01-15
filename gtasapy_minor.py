import numpy as np
import os
import six.moves.urllib as urllib
import sys
sys.path.insert(0, "E:\workspace_py\object_detection")
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO

from matplotlib import pyplot as plt
from PIL import Image, ImageGrab
import cv2
# import pyHook, pythoncom
from directKeys import PressKey, ReleaseKey, W, A, S, D, J
# cap = cv2.VideoCapture(0)


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

# Detection
real_height = 420.0 # experimental value
focal_length = 10.0 # experimental value
brain_ON = True
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      # ret, image_np = cap.read()

      game_screen = np.asarray(ImageGrab.grab(bbox=(1, 35, 640, 420)))
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
      # Actual detection.
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
      apparent_height = (box_with_max_score[2] - box_with_max_score[0])*420.0 # multiplying to de-normalize.
      print("\nApparent height = ", apparent_height)
      print("\nApparent height (normalized) = ", apparent_height / 420.0)
      distance = ((real_height*focal_length)/apparent_height) - focal_length
      # velocity = (distance - )/5
      print("\nDistance from object = ", distance)

      # if brain_ON:
      # 	if distance < 40.0:
      # 		ReleaseKey(W)
      # 		PressKey(S)
      # 	else:
      # 		PressKey(W)
      # 		ReleaseKey(S)


      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)

      cv2.imshow('object detection', cv2.resize(image_np, (640,420)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
      elif cv2.waitKey(25) & 0xFF == ord('b'):
      	brain_ON = not brain_ON
      	print("b pressed.")

# cap.close()