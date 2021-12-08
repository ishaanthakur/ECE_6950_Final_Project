
from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
import numpy as np
import cv2

def detect_people(frame, net, ln, personIdx=0):
	
	(H, W) = frame.shape[:2]
	res = []

	
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False) # blob construct from input frame
	net.setInput(blob) 
	layerOutputs = net.forward(ln) # forard pass of Yolofor getting bounding boxes

	
	boxes = [] # list for bb
	centroids = [] # list for centroid
	confidences = [] # list for confidence


	
	for out in layerOutputs: #loop over outputs
		for det in out: # loop over detections
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = det[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

	
			if classID == personIdx and confidence > MIN_CONF: # person detected and confidence high enough

				
				box = det[0:4] * np.array([W, H, W, H]) #scaling bb relative to size of image
				(centerX, centerY, width, height) = box.astype("int")

		
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	
	id_detect = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH) # spurress weak overlapping bb

	
	if len(id_detect) > 0: # if one detection is there
		
		for i in id_detect.flatten():
			
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			res.append(r)

	return res