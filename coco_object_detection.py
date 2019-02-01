# cannot keep file name as object_detection.py cause object_detection folder already exists and it confuses python imports.

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
# from directKeys import PressKey, ReleaseKey, W, A, S, D, J
# import pyHook, pythoncom

# Object detection imports
# Here are the imports from the object detection module.
# utils is folder in object_detection/

from utils import label_map_util

from utils import visualization_utils as vis_util

class ObjectDetection:
	def __init__(self):
		## Variables
		# 
		# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
		# 
		# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

		# What model to download.
		self.MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017' # model name.
		self.MODEL_FILE = self.MODEL_NAME + '.tar.gz'
		self.DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

		# Path to frozen detection graph. This is the actual model that is used for the object detection.
		self.PATH_TO_CKPT = self.MODEL_NAME + '/frozen_inference_graph.pb'

		# List of the strings that is used to add correct label for each box.
		self.PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')

		self.NUM_CLASSES = 90

	def download_model(self):
		# Download Model
		opener = urllib.request.URLopener()
		opener.retrieve(self.DOWNLOAD_BASE + self.MODEL_FILE, self.MODEL_FILE)
		tar_file = tarfile.open(self.MODEL_FILE)
		for file in tar_file.getmembers():
			file_name = os.path.basename(file.name)
			if 'frozen_inference_graph.pb' in file_name:
				tar_file.extract(file, os.getcwd())

	def load_model(self):
		# Load a (frozen) Tensorflow model into memory.

		# It's unnecessary, but it's a good practice to follow.
		# Since a default graph is always registered, every op and variable is placed into the default graph.
		# The statement, however, creates a new graph and places everything (declared inside its scope) into this graph.
		# If the graph is the only graph, it's useless. But it's a good practice.

		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			self.od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid: 
				serialized_graph = fid.read()
				self.od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(self.od_graph_def, name='')

	def init_label_map(self):
		# Loading label map
		# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.
		# Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

		self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
		self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
		self.category_index = label_map_util.create_category_index(self.categories)
		# print("categories: ", self.categories)
		# print("\n\ncategory_index:", self.category_index)

	def initialize(self, download_model=False):
		if download_model:
			self.download_model()
		self.load_model()
		self.init_label_map()


		self.detection_graph.as_default()
		self.sess = tf.Session(graph = self.detection_graph)
		print("\nModel initialized.")

	def do_inference(self, image_np, return_visualized_image=False):
		self.image = image_np
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
		# Each box represents a part of the image where a particular object was detected.
		boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
		classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
		# Actual detection (inference).
		(self.boxes, self.scores, self.classes, self.num_detections) = self.sess.run(
			[boxes, scores, classes, num_detections],
			feed_dict={image_tensor: image_np_expanded})

		if return_visualized_image:
			return self.boxes, self.scores, self.classes, self.num_detections, self.get_visualized_image()
		else:
			return self.boxes, self.scores, self.classes, self.num_detections

	def get_visualized_image(self):
		try:
			vis_util.visualize_boxes_and_labels_on_image_array(
				  self.image,
				  np.squeeze(self.boxes),
				  np.squeeze(self.classes).astype(np.int32),
				  np.squeeze(self.scores),
				  self.category_index,
				  use_normalized_coordinates=True,
				  line_thickness=2)
			return self.image
		except:
			pass

	def destruct(self):
		self.sess.close()

##################################
############## Test ##############
##################################

# objectDetection = ObjectDetection()
# objectDetection.initialize()

# from cv2 import imread
# import matplotlib.pyplot as plt

# img = imread("E:/workspace_py/Japanese_spaniel/j31.jpg")
# x, y, z, t, img = objectDetection.do_inference(img, True)

# plt.imshow(img)
# plt.show()