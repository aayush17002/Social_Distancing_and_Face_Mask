import os
import numpy as np
import face_detection 
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import tqdm
import sys
import shutil
import time
import argparse
from datetime import date
import cv2
import math
from scipy.spatial import distance as dist

detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

mask_classifier = load_model("ResNet50_Classifier2.h5")

# cap = cv2.VideoCapture('with_without_masks.mp4')
cap = cv2.VideoCapture('vid1.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(fps,width,height,n_frames)

net = cv2.dnn.readNet("Models/"+"yolov4_tiny.weights", "Models/"+"yolov4_tiny.cfg")

classes = []
with open("Models/"+"coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

print("now starting")

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('with_without_masks.mp4')
# cap = cv2.VideoCapture('vid1.mp4')


fps = cap.get(cv2.CAP_PROP_FPS)
# WW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# HH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# print(fps,WW,HH,n_frames)

v=fps//5+1

try:  
    os.mkdir("Results")  
except OSError as error:  
    shutil.rmtree( "Results")
    os.mkdir( "Results")

try:  
    os.mkdir("Results/Frames")  
except OSError as error:  
    shutil.rmtree("Results/Frames")
    os.mkdir("Results/Frames")


def draw_prediction(frame, classes, classId, conf, left, top, right, bottom,color,temp,L,i):
    
    label = str(i)  

    if classes:
        assert(classId < len(classes))
    cv2.rectangle(frame, (left, top), (right, bottom), color,2)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.6,color,1)
    

    if temp==False:
        text="Number of Social Distancing Violations= "+str(L)
        cv2.putText(frame,text,(15,frame.shape[0]-15),cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,255,0), 1)


c1 = 0
# for frame in tqdm.notebook.tqdm(range(int(n_frames))):
while(cap.isOpened()):
    t0 = time.time()    
    ret, img = cap.read()

    if ret == False:
        break;
    if c1%v==0:
        HH, WW, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)


        frameHeight = img.shape[0]
        frameWidth = img.shape[1]
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

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        violate=set()

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

                            if score[0][0]<=0.5:
                                masked_faces.append([x1,y1,x2,y2])
                            else:
                                unmasked_faces.append([x1,y1,x2,y2])

                        except:
                            continue
                except:
                    continue
                        
        masked_face_count = len(masked_faces)
        unmasked_face_count = len(unmasked_faces)
        
        for f in range(masked_face_count):
            a,b,c,d = masked_faces[f]
            cv2.rectangle(img, (a, b), (c,d), (0,255,0), 2)

        for f in range(unmasked_face_count):
            a,b,c,d = unmasked_faces[f]
            cv2.rectangle(img, (a, b), (c,d), (0,0,255), 2)

        HH, WW, channels = img.shape

        cv2.rectangle(img,(0,0),(WW,20),(0,0,0),-1)
        cv2.rectangle(img,(1,1),(WW-1,20),(255,255,255),2)

        xpos = 15

        string = "Total People = "+str(len(results))
        cv2.putText(img,string,(xpos,35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
        xpos += cv2.getTextSize(string,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)[0][0]

        string = " ( "+str(len(results)-len(violate)) + " Safe "
        cv2.putText(img,string,(xpos,35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        xpos += cv2.getTextSize(string,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)[0][0]

        string = str(len(violate))+ " Unsafe ) "
        cv2.putText(img,string,(xpos,35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        xpos += cv2.getTextSize(string,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)[0][0]

        string = "( " +str(masked_face_count)+" Masked "+str(unmasked_face_count)+" Unmasked )"
        cv2.putText(img,string,(xpos,35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)



        temp=False
        if len(results)>=2:

            c=np.array([r[2] for r in results ])
            d=dist.cdist(c, c, metric="euclidean")
            distances_list=[]

            for i in range(d.shape[0]):
                for j in range(i+1,d.shape[1]):

                    h_=(heights_[i]+heights_[j])/2
                    val=6*(h_/5.5)

                    if d[i,j]<val:
                        cv2.line(img,results[i][2],results[j][2],(255,255,255),1)
                        violate.add(i)
                        violate.add(j)
                        DIST=max(0,round( ((6/val)*(d[i,j])  ),2))
                        distances_list.append([i,j,DIST])

            
            for (i,(ind,bbox,center)) in enumerate(results):
                (startx,starty,endx,endy)=bbox
                (cx,cy)=center
                color=(0,255,0)

                if i in violate:
                    color=(0,0,255)
                draw_prediction(img, classes, classIds[ind], confidences[ind], startx, starty, endx,endy,color,temp,len(violate),i)
                temp=True

            for xx in range(len(distances_list)):
                label=("Distance from "+str(distances_list[xx][0])+" to "+ str(distances_list[xx][1])+" = "+str(distances_list[xx][2])+"ft")
                cv2.putText(img,label,(15,20+(12*xx)),cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,0), 1)
                


        # return len(violate)     



        cv2.imshow("Facemask Detection", img)  
        if len(violate)>0:
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
            img_name=date_+"_"+ttt+"_"+str(len(violate))+".png"
            # cv2.imshow("kaka", img)  

            cv2.imwrite( "Results/Frames/"+img_name+".jpg",img)
            



        t1 = time.time()
        total = t1-t0
        print("Total time taken= ",total)
        
        key=cv2.waitKey(5)
        if key==27:
            break
    c1+=1


# out_stream.release()
cap.release()
cv2.destroyAllWindows()

print("Done !")
