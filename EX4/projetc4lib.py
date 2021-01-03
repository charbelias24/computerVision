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
Rectangle = namedtuple('Rectangle', ['x1', 'y1', 'x2', 'y2'])



def sliding_window(image, rectangleList, stepSize, windowSize):
  for rectangle in rectangleList:
    for y in range(rectangle.y1,rectangle.y2 - windowSize[0],stepSize):
      for x in range(rectangle.x1,rectangle.x2 - windowSize[0],stepSize):
        yield(x,y,image[y:y+windowSize[1],x:x+windowSize[0]])

def predict_sliding_window_cnn(frame,ROIrectangle,net,average):
  predictions = []
  positive_windows = []
  total_windows = 0
  window_sizes=[]
  ## let's resize the ROI to shrink it a lot more:
  width = ROIrectangle.x2 - ROIrectangle.x1
  height = ROIrectangle.y2 - ROIrectangle.y1
  rectangleList = []
  if(width <= 295) | (int(average) <= 250):
    rectangle = Rectangle(ROIrectangle.x1 + width//4 ,ROIrectangle.y1,ROIrectangle.x2 - width//4, ROIrectangle.y1 + height//3)
    rectangleList.append(rectangle)
  else:
    rectangle1 = Rectangle(ROIrectangle.x1 + width//8, ROIrectangle.y1 + height//10, ROIrectangle.x1 + (3*width)//7, ROIrectangle.y1 + height//3)
    rectangle2 = Rectangle(ROIrectangle.x2 - (3*width)//7, ROIrectangle.y1 + height//10, ROIrectangle.x2 - width//8, ROIrectangle.y1 + height//3)
    rectangleList.append(rectangle1)
    rectangleList.append(rectangle2)
  for rectangle in rectangleList:
    #cv2.rectangle(frame,(rectangle.x1,rectangle.y1),(rectangle.x2, rectangle.y2), (0,0,255),1)
    window_sizes.append(int((rectangle.x2 - rectangle.x1)/2))
  for window_size in window_sizes:
    for i,(x,y,window) in enumerate(sliding_window(frame,rectangleList,window_size//4,(window_size,window_size))):
      if window.shape[0] != window_size or window.shape[1] != window_size:
        continue
      # Preprocessing the window:
      total_windows+=1
      window_p = keras.preprocessing.image.smart_resize(window,size=(16,16)) 
      window_array = keras.preprocessing.image.img_to_array(window_p)
      window_array = tf.expand_dims(window_array, 0) # batch axis
      (back,face) = net(window_array)[0]
      if (face > back): # face > back
        text = "{:.2f}%".format(face * 100)
        positive_windows.append(Rectangle(x,y,x+window_size,y+window_size))
        color = (255,0,0)
        cv2.rectangle(frame,(x,y),(x+window_size, y + window_size), color,2)
        cv2.putText(frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2) 
  return frame, total_windows, positive_windows

def predict_pretrained_ROI(frame, ROIrectangle, net, ROI_strategy = True):
    positive_windows = []
    if (ROI_strategy):
        ROI = frame[ROIrectangle.y1:ROIrectangle.y2,ROIrectangle.x1:ROIrectangle.x2]
    else:
        ROI = frame
    (h,w) = ROI.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(ROI, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0,detections.shape[2]):
        #clone = ROI.copy()
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
        color = (255,0,0)
        ## find box in frame, not in ROI:
        x1f = ROIrectangle.x1 + startX
        y1f = ROIrectangle.y1 + startY
        x2f = ROIrectangle.x1 + endX
        y2f = ROIrectangle.y1 + endY
        cv2.rectangle(frame, (x1f, y1f), (x2f, y2f),
            color, 2)
        positive_windows.append(Rectangle(x1f,y1f,x2f,y2f))
        cv2.putText(frame, text, (x1f, y1f),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2) 
    return frame, positive_windows

def draw_truth(frame, list_faces):
    color = (0,255,0)
    for face_rec in list_faces:
        cv2.rectangle(frame, (int(face_rec.x1), int(face_rec.y1)), (int(face_rec.x2), int(face_rec.y2)), color, 2)
    return frame

def calculate_area(rect: Rectangle):
    return (rect.x2 - rect.x1) * (rect.y2 - rect.y1)

def check_overlap(rect1: Rectangle, rect2: Rectangle):      
    # If one rectangle is on left side of other 
    if rect1.x1 >= rect2.x2 or rect1.x2 <= rect2.x1:
        return False
  
    # If one rectangle is above other 
    if rect1.y1 >= rect2.y2 or rect1.y2 <= rect2.y1:
        return False
  
    return True

def calculate_IOU(rect1: Rectangle, rect2: Rectangle):
    if not check_overlap(rect1, rect2):
        return 0
    
    area1 = calculate_area(rect1)
    area2 = calculate_area(rect2)
    
    rect_intersection = Rectangle(
        max(rect1.x1, rect2.x1),
        max(rect1.y1, rect2.y1),
        min(rect1.x2, rect2.x2),
        min(rect1.y2, rect2.y2)
    )
    
    area_intersection = calculate_area(rect_intersection)

    return area_intersection / (area1 + area2 - area_intersection)

def parse_ground_truth(gt_file, last_frame):
    dict_faces = {}
    positive_windows = {}
    for i in range(last_frame+1):
        dict_faces[i] = []
        positive_windows[i] = []
    fd = open(gt_file, 'r')
    while True: 
        line = fd.readline().rstrip()
        if not line:
            break
        line_list = line.split(',')
        frame_nb = int(line_list[0])
        x1 = float(line_list[2])
        y1 = float(line_list[3])
        x2 = x1 + float(line_list[4])
        y2 = y1 + float(line_list[5])
        dict_faces[frame_nb].append(Rectangle(x1,y1,x2,y2))
    fd.close()
    return dict_faces, positive_windows

def compute_metrics(list_IOU, unmatched_faces):
    FP , TP = check_positives(list_IOU)
    FN = check_negatives(unmatched_faces)
    return (TP)/(TP+FP), (TP)/(TP+FN)

def get_list_IOU(positive_windows,dict_faces):
  list_IOU = {}
  unmatched_faces = {}
  for frame_nb,window_list in positive_windows.items():
    list_IOU[frame_nb] = []
    unmatched_faces[frame_nb] = []
    faces_in_image = dict_faces[frame_nb]
    if len(window_list) > 0:
        for predicted_rec in window_list:
            list_IOU_rec = []
            for truth_rec in faces_in_image:
                IOU = calculate_IOU(predicted_rec,truth_rec)
                if IOU < 0.3:
                    unmatched_faces[frame_nb].append(truth_rec)
                else:
                    if truth_rec in unmatched_faces[frame_nb]:
                        unmatched_faces[frame_nb].remove(truth_rec)
                list_IOU_rec.append(IOU)
            r = True
            for IOU in list_IOU_rec:
                r = r & bool(IOU)
            if r is False:
                list_IOU_rec = [i for i in list_IOU_rec if i != 0]
            list_IOU[frame_nb]+=list_IOU_rec
    else:
        if (len(faces_in_image) != 0):
            for truth_rec in faces_in_image:
                unmatched_faces[frame_nb].append(truth_rec)
    list_IOU[frame_nb] = list(dict.fromkeys(list_IOU[frame_nb]))
    unmatched_faces[frame_nb] = list(dict.fromkeys(unmatched_faces[frame_nb]))
  return list_IOU, unmatched_faces


def check_negatives(unmatched_faces):
    return sum([len(unmatched_faces[x]) for x in unmatched_faces])

def check_positives(list_IOU):
  false_positives = 0
  true_positives = 0
  for frame_nb,IOU_list in list_IOU.items():
      for IOU in IOU_list:
          if IOU > 0.3:
              true_positives +=1
          elif(IOU != -1):
              false_positives+=1
  return false_positives, true_positives
