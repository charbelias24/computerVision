from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import imutils
from imutils import paths
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from imutils.video import VideoStream
import time
import argparse
from collections import namedtuple
import projetc4lib as lib
from pathlib import Path
from imutils.video import count_frames

## AVDIAR_All/Seq01-1P-S0M1/Video/Seq01-1P-S0M1_CAM1.mp4
## AVDIAR_All/Seq02-1P-S0M1/Video/Seq02-1P-S0M1_CAM1.mp4
## AVDIAR_All/Seq03-1P-S0M1/Video/Seq03-1P-S0M1_CAM1.mp4
## AVDIAR_All/Seq04-1P-S0M1/Video/Seq04-1P-S0M1_CAM1.mp4
## AVDIAR_All/Seq05-2P-S1M0/Video/Seq05-2P-S1M0_CAM1.mp4
## AVDIAR_All/Seq06-2P-S1M0/Video/Seq06-2P-S1M0_CAM1.mp4
## AVDIAR_All/Seq07-2P-S1M0/Video/Seq07-2P-S1M0_CAM1.mp4
## AVDIAR_All/Seq08-3P-S1M1/Video/Seq08-3P-S1M1_CAM1.mp4
## AVDIAR_All/Seq40-2P-S2M0/Video/Seq40-2P-S2M0_CAM1.mp4

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")                        
args = vars(ap.parse_args())
Rectangle = namedtuple('Rectangle', ['x1', 'y1', 'x2', 'y2'])
netPreMade = cv2.dnn.readNetFromCaffe('./deploy.prototxt','./res10_300x300_ssd_iter_140000.caffemodel')
## cnn:
json_file = open('./cnn_WIDER.json','r')
loaded_cnn_json = json_file.read()
json_file.close()
cnn = keras.models.model_from_json(loaded_cnn_json)
cnn.load_weights('./cnn_WIDER.h5')
opt = keras.optimizers.Adam(lr=1e-3,decay=1e-3/25)
cnn.compile(loss="binary_crossentropy", optimizer = opt, metrics=['accuracy',
                                                                         keras.metrics.Precision(),
                                                                         keras.metrics.Recall(),
                                                                         keras.metrics.AUC()
])
precision_list = []
for video in args["video"].split(','):
  print('#######################')
  print("starting video "+video)
  vs = cv2.VideoCapture(video)
  path = Path(video)
  gt_file = path.parent.parent
  gt_file = str(gt_file) + '/GroundTruth/face_bb.txt'
  last_frame = count_frames(video) - 1
  print("Last frame of video = "+str(last_frame))
  dict_faces, positive_windows = lib.parse_ground_truth(gt_file,last_frame)
  firstFrame = None
  subtractor = cv2.createBackgroundSubtractorMOG2(history = 50, varThreshold= 200, detectShadows = True)
  width_avg_list = []
  frame_nb = -1
  while True:
  # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
      frame = vs.read()
      frame = frame[1]  
      if frame is None:
          precision_video = lib.compute_precision(positive_windows,dict_faces)
          print("The precision of the method in the video: "+str(video)+ " is " +str(precision_video))
          precision_list.append(precision_video)
          break
      frame_nb+=1
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
          text = 'ROI'
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
          width_avg_list.append(w)
          average = sum(width_avg_list) / len(width_avg_list)
          # print('ROI width = '+ str(w))
          # print('ROI width AVERAGE = '+ str(average))
          cv2.putText(frame, text, (x, y),
          cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2) 
          ROI = Rectangle(x,y,x+w,y+h)
          if (ROI.x2 == 720) & (ROI.y2 == 450):
            continue
          #clone, positive_windows_list = lib.predict_pretrained_ROI(frame, ROI, netPreMade)
          clone, total_windows, positive_windows_list = lib.predict_sliding_window_cnn(frame, ROI,cnn,average)
          positive_windows[frame_nb]+= positive_windows_list
          if (clone is not None):
            clone = lib.draw_truth(clone,dict_faces[frame_nb])
            cv2.imshow("frame",clone if not None else frame)
          else:
            frame = lib.draw_truth(frame,dict_faces[frame_nb])
            cv2.imshow("frame",frame)
          key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
          if key == ord("q"):
              break
  vs.stop() if args.get("video", None) is None else vs.release()
  cv2.destroyAllWindows()
print("the total precision is = "+str(sum(precision_list)/len(precision_list)))