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
if not args.get("video", False):
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    vs = cv2.VideoCapture(args["video"])
firstFrame = None
subtractor = cv2.createBackgroundSubtractorMOG2(history = 50, varThreshold= 200, detectShadows = True)
# loop over the frames from the video stream
while True:
# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    frame = vs.read()
    frame = frame[1] if args.get("video",False) else frame 
    text = "Unoccupied"
    if frame is None:
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    subs = subtractor.apply(gray, learningRate = 0)
    ## preprocessing
    alpha = 255/frame.max()
    ## adjust brightness:
    bright = cv2.convertScaleAbs(subs, alpha = alpha, beta = 0)
    _, thresh = cv2.threshold(bright,125,255,0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh, kernel, iterations=10)
    cnts = cv2.findContours(thresh.astype(np.uint8).copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
	## dont grab every contour, only bigger than a minimun area (to avoid minimun changes to bo accounted)
        if cv2.contourArea(c) < 12000:
            continue
  ## show results:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #(x, y, w, h) = cv2.boundingRect(cnts)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Subs", subs)
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()