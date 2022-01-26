import cv2
import numpy as np
import math
from scipy.spatial import distance as dist

def draw_prediction(frame, classes, classId, conf, left, top, right, bottom,color,temp,L,i):
    
    label = str(i)  

    if classes:
        assert(classId < len(classes))
    cv2.rectangle(frame, (left, top), (right, bottom), color,2)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.6,color,1)
    

    if temp==False:
        text="Number of Social Distancing Violations= "+str(L)
        cv2.putText(frame,text,(15,frame.shape[0]-15),cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,255,0), 1)
        


def process_frame(frame, outs, classes, confThreshold, nmsThreshold):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height,center_x,center_y])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    violate=set()

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

    # avgg=sum(heights_)/len(heights_)

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
                    cv2.line(frame,results[i][2],results[j][2],(255,255,255),1)
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
            draw_prediction(frame, classes, classIds[ind], confidences[ind], startx, starty, endx,endy,color,temp,len(violate),i)
            temp=True

        for xx in range(len(distances_list)):
            label=("Distance from "+str(distances_list[xx][0])+" to "+ str(distances_list[xx][1])+" = "+str(distances_list[xx][2])+"ft")
            cv2.putText(frame,label,(15,20+(15*xx)),cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,0), 1)
            


    return len(violate)


    

# h_ -> 5.5
# 1 -> 5.5/h_







