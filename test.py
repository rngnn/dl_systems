# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os


class BoundingBox:
	def __init__(self, start_x, start_y, end_x, end_y):
		self.start_x = start_x
		self.start_y = start_y
		self.end_x = end_x
		self.end_y = end_y


class TestCase:
	def __init__(self, detections, confidence, all_detections, detections_with_confidence):
		self.detections = detections
		self.confidence = confidence
		self.all_detections = all_detections
		self.detections_with_confidence = detections_with_confidence



def mask_image():
	IMAGE_PATH = './images/pic1.jpeg'
	FACE_DETECTOR_DIR_PATH = 'face_detector'
	MODEL_PATH = 'mask_detector.model'
	CONFIDENCE = 0.5

	TEST_CASE = TestCase(detections=[BoundingBox(504, 204, 650, 394), BoundingBox(169, 191, 341, 436)],
						 confidence=CONFIDENCE,
						 all_detections=200,
						 detections_with_confidence=2)


	prototxtPath = os.path.sep.join([FACE_DETECTOR_DIR_PATH, "deploy.prototxt"])
	weightsPath = os.path.sep.join([FACE_DETECTOR_DIR_PATH, "res10_300x300_ssd_iter_140000.caffemodel"])
	net = cv2.dnn.readNet(prototxtPath, weightsPath)

	model = load_model(MODEL_PATH)

	image = cv2.imread(IMAGE_PATH)
	orig = image.copy()
	(h, w) = image.shape[:2]

	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	net.setInput(blob)
	detections = net.forward()
	all_detections_number = detections.shape[2]
	assert all_detections_number == TEST_CASE.all_detections

	result_detections_with_confidence = 0
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > CONFIDENCE:
			result_detections_with_confidence += 1

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			assert TEST_CASE.detections[i].start_x == startX
			assert TEST_CASE.detections[i].start_y == startY
			assert TEST_CASE.detections[i].end_x == endX
			assert TEST_CASE.detections[i].end_y == endY

	assert result_detections_with_confidence == TEST_CASE.detections_with_confidence


	print('TEST CASE PASSED')
	
if __name__ == "__main__":
	mask_image()
