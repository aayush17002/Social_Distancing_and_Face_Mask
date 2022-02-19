import cv2
import numpy as np
from utils import process_frame, draw_prediction
import os
import socket
import pickle

def connections():
    PORT1 = 8500
    PORT2 = 8400
    #Connect ot the server
    server_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_conn.connect(('192.168.1.170', PORT1))
    print("Input is now connected to server")
    #Connect ot the final output
    output_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    output_conn.connect(('192.168.20.238', PORT2))
    print("Input is now connected to output")

    return server_conn, output_conn

def video_initialization():
    capture = cv2.VideoCapture(0)

    capture.set(3, 320);
    capture.set(4, 240);
    # cap = cv2.VideoCapture('Data/sd2.mp4')
    # cap = cv2.VideoCapture('Data/violence.mp4')

    fps = capture.get(cv2.CAP_PROP_FPS)
    print("FPS= ",fps)
    step=fps//5+1

    return capture, step

def load_model():
    with open('Models/coco.names', 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    net = cv2.dnn.readNet(os.path.abspath("Models/yolov4_tiny.weights"), os.path.abspath("Models/yolov4_tiny.cfg"), "darknet")
    outNames = net.getUnconnectedOutLayersNames()
    return classes, net, outNames

def send_data_server(frame, server_conn, HEADERSIZE):
    data = pickle.dumps(frame)
    msg = bytes(f"{len(data):<{HEADERSIZE}}", 'utf-8')+data
    server_conn.send(msg)
    return len(data)

def send_social_distancing(frame, classes, net, outNames, output_conn, CONF_THRESHOLD, NMS_THRESHOLD):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(outNames)
    violation_count=process_frame(frame, outs, classes, CONF_THRESHOLD, NMS_THRESHOLD)
    
    msg = "0"
    if violation_count>0:
        msg = "1"
    output_conn.send(bytes(msg,"utf-8"))

#initialise varibales
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
img_counter = 0
HEADERSIZE = 10
count = 0

#Load an initialise
classes, net, outNames = load_model()
server_conn, output_conn = connections()
capture, step = video_initialization()

while(cap.isOpened()):
    ret, frame = capture.read()

    if not ret:
        break;
    
    if count%step==0:
        size = send_data_server(frame, server_conn, HEADERSIZE)
        print("{}: {}".format(img_counter, size))
        send_social_distancing(frame, classes, net, outNames, output_conn, CONF_THRESHOLD, NMS_THRESHOLD)
        img_counter += 1    

    count+=1

capture.release()
cv2.destroyAllWindows()
