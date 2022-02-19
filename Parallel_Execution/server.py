import socket
import pickle
import numpy as np
import os
import cv2
import time
from datetime import date
import face_detection 
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

def get_models():
	detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

	mask_classifier = load_model("ResNet50_Classifier2.h5")
	net = cv2.dnn.readNet("Models/"+"yolov4_tiny.weights", "Models/"+"yolov4_tiny.cfg")
	classes = []
	with open("Models/"+"coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
		
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	return detector, mask_classifier, classes, output_layers, net

def create_server():
	PORT1 = 8500
	PORT2 = 8450
	#Connect ot the final output
	output_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	output_conn.connect((socket.gethostname(), PORT2))
	print('Server is now connected to output')
	#Bind the server
	server_conn=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	server_conn.bind((socket.gethostname(),PORT1))
	server_conn.listen(5)
	print('Server is now listening to input')
	return server_conn, output_conn

def get_message(sock):
	HEADERSIZE = 10
	full_msg = b''
	new_msg = True
	while True:
		msg = sock.recv(4096)
		if new_msg:
			print("new msg len:",msg[:HEADERSIZE])
			msglen = int(msg[:HEADERSIZE])
			new_msg = False

		full_msg += msg

		if len(full_msg)-HEADERSIZE == msglen:
			print("full msg recvd")
			return pickle.loads(full_msg[HEADERSIZE:])

def get_image_pred(conn,net,output_layers):
	img = get_message(conn)
	HH, WW, channels = img.shape
	blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)
	return img, outs

def get_bounding_info(img_shape,outs):
	frameHeight = img_shape[0]
	frameWidth = img_shape[1]
	classIds = []
	confidences = []
	boxes = []

	for out in outs:
		for detection in out:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]
			if confidence > 0.5:
				center_x = int(detection[0] * frameWidth)
				center_y = int(detection[1] * frameHeight)
				width = int(detection[2] * frameWidth)
				height = int(detection[3] * frameHeight)
				left = int(center_x - width / 2)
				top = int(center_y - height / 2)
				classIds.append(classId)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height,center_x,center_y])
	return classIds, boxes, confidences

def mask_info(indices,classes,classIds,boxes,detector,img,mask_classifier):
	mask_violation = False

	masked_faces = []
	unmasked_faces = []

	heights_=[]
	widths_=[]
	results=[]

	for i in indices:
		i = i[0]
		if classes[classIds[i]]=="person":
			box = boxes[i]
			x = box[0]
			y = box[1]
			width = box[2]
			height = box[3]
			heights_.append(height)
			widths_.append(width)
			centerx=int(x+width/2)
			centery=int(y+height/2)
			r=(i,(x,y,x+width,y+height),(centerx,centery))
			results.append(r)

			person_rgb = img[y:y+height,x:x+width,::-1] 
			
			try:
				detections = detector.detect(person_rgb)

				if detections.shape[0] > 0:

					detection = np.array(detections[0])
					detection = np.where(detection<0,0,detection)

					x1 = x + int(detection[0])
					x2 = x + int(detection[2])
					y1 = y + int(detection[1])
					y2 = y + int(detection[3])

					try :
						face_rgb = img[y1:y2,x1:x2,::-1]   
						face_arr = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
						face_arr = np.expand_dims(face_arr, axis=0)
						face_arr = preprocess_input(face_arr)
						score = mask_classifier.predict(face_arr)

						if score[0][0]>0.5:
							mask_violation = True
							break
					except:
						continue
			except:
				continue
	return mask_violation

def save_img(img):
	today = date.today()
	t = time.localtime()
	date_=today.strftime("%b-%d-%Y")
	current_time = time.strftime("%H:%M:%S", t)
	ttt=""
	for q in range(len(current_time)):
		if current_time[q]==":":
			ttt+="-"
		else:
			ttt+=current_time[q]
	img_name=date_+"_"+ttt
	imgx = cv2.resize(img,(16*40,9*40))
	cv2.imwrite("Results/Input_Frames/"+img_name+".jpg",imgx)

detector, mask_classifier, classes, output_layers, net = get_models()
print("now starting")

server_conn, output_conn = create_server()
conn,addr=server_conn.accept()
print('Connected to input by', addr)

while True:	
	img, outs = get_image_pred(conn,net,output_layers)
	classIds, boxes, confidences = get_bounding_info(img.shape,outs)
	indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	if mask_info(indices,classes,classIds,boxes,detector,img,mask_classifier):
		output_conn.send(bytes("1","utf-8"))
	else:
		output_conn.send(bytes("0","utf-8"))
	save_img(img)
