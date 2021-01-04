import keras
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import imutils
from imutils import paths
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from glob import glob
from imutils.video import VideoStream
import time
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")                        
args = vars(ap.parse_args())
    
## video: AVDIAR_All/Seq01-1P-S0M1/Video/Seq01-1P-S0M1_CAM1.mp4
## AVDIAR_All/Seq03-1P-S0M1/Video/Seq03-1P-S0M1_CAM1.mp4
net = cv2.dnn.readNetFromCaffe('./deploy.prototxt','./res10_300x300_ssd_iter_140000.caffemodel')
#net = cv2.dnn.readNetFromTensorflow('./optmized_graph.pb','./optmized_graph.pbtxt')
if not args.get("video", False):
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    vs = cv2.VideoCapture(args["video"])

# loop over the frames from the video stream
while True:
# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
  frame = vs.read()
  frame = frame[1] if args.get("video",False) else frame 
  if frame is None:
    break
  #frame = imutils.resize(frame, width=600)
    # grab the frame dimensions and convert it to a blob
  (h, w) = frame.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
      (300, 300), (104.0, 177.0, 123.0))
  
    # pass the blob through the network and obtain the detections and
    # predictions
  net.setInput(blob)
  detections = net.forward()

  for i in range(0,detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]
    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence < 0.5:
      continue
    # compute the (x, y)-coordinates of the bounding box for the
    # object
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    
        # draw the bounding box of the face along with the associated
        # probability
    text = "{:.2f}%".format(confidence * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(frame, (startX, startY), (endX, endY),
          (0, 0, 255), 2)
    cv2.putText(frame, text, (startX, y),
    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)  

  


  cv2.imshow("Video",frame)
  key = cv2.waitKey(1) & 0xFF