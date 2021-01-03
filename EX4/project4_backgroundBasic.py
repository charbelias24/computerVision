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
## AVDIAR_All/Seq07-2P-S1M0/Video/Seq07-2P-S1M0_CAM1.mp4
## AVDIAR_All/Seq05-2P-S1M0/Video/Seq05-2P-S1M0_CAM1.mp4
if not args.get("video", False):
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    vs = cv2.VideoCapture(args["video"])
firstFrame = None
# loop over the frames from the video stream
while True:
# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    frame = vs.read()
    frame = frame[1] if args.get("video",False) else frame 
    text = "Unoccupied"
    if frame is None:
        break
    frame = imutils.resize(frame, width=500)
 ## work with grayscale first:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 ## blurr the frame:
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
 ## first frame init:
    if firstFrame is None:
        firstFrame = gray
        continue
  ## difference between current frame and first frame:
    frameDelta = cv2.absdiff(firstFrame, gray)
 ## now lets asign either a 0 or a 255 to each pixel: if the delta is greater than 25 then we'll set it to a person
    thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]
 ## dilate the thresholded image to fill in holes, then find contours
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
	## dont grab every contour, only bigger than a minimun area (to avoid minimun changes to bo accounted)
        if cv2.contourArea(c) < 2000:
            continue
  ## show results:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
  #cv2_imshow(frame)
    cv2.imshow('frame',frame)
    cv2.imshow("Thresh", thresh)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()