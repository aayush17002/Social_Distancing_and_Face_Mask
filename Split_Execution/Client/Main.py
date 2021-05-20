import cv2
import numpy as np
import argparse
import os
from utils import process_frame, draw_prediction
import time
import math
from datetime import date
import os
import sys
import shutil
import io
import socket
import struct
import time
import pickle
import zlib

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 8485))
connection = client_socket.makefile('wb')


CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4

with open('Models/coco.names', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNet(os.path.abspath("Models/yolov4_tiny.weights"), os.path.abspath("Models/yolov4_tiny.cfg"), "darknet")
outNames = net.getUnconnectedOutLayersNames()
writer = None

cap = cv2.VideoCapture(0)

cap.set(3, 320);
cap.set(4, 240);
# cap = cv2.VideoCapture('Data/sd2.mp4')
# cap = cv2.VideoCapture('Data/violence.mp4')

count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS= ",fps)
v=fps//5+1

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while(cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        break;
    
    if(count%(v)==0):
        frame_send = frame.copy()
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(outNames)
        L=process_frame(frame, outs, classes, CONF_THRESHOLD, NMS_THRESHOLD)
        
        # social distancing not followed
        if L>0:
            result, frame = cv2.imencode('.jpg', frame_send, encode_param)
        #    data = zlib.compress(pickle.dumps(frame, 0))
            data = pickle.dumps(frame, 0)
            size = len(data)

            print("{}: {}".format(img_counter, size))
            client_socket.sendall(struct.pack(">L", size) + data)
            img_counter += 1

    count+=1

cap.release()
cv2.destroyAllWindows()