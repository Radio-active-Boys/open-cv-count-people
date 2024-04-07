import cv2 as cv
import torch
from tracker import *
import numpy as np
import pandas as pd
from ultralytics import YOLO
# load model
model=YOLO('yolov8s.pt')

cap=cv.VideoCapture('./vidp/cctv.mp4')

# get mouse corrdinates
def Coordinates(event,x,y,flags,param):
    if event == cv.EVENT_MOUSEMOVE :
        colorsBGR = [x,y]
        print(colorsBGR)
        
cv.namedWindow('FRAME')
cv.setMouseCallback('FRAME', Coordinates)


# coco text
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

# tracker  initialization

tracker = Tracker()
while True:
    ret,frame=cap.read()
    frame=cv.resize(frame,(1020,500))
    results = model(frame)
    # print("********************")
    # print(results)
    # result = model.predict(frame)
    # print(results)
    # print("********************")
    a = results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
     
    list=[]
   
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        if 'person' in c:

            list.append([x1,y1,x2,y2])
    
    bboxes_id = tracker.update(list)
    
    new_ids = set()
    for bbox in bboxes_id:
        x1,y1,x2,y2,id = bbox
        center_x = int(x1+x2)//2
        center_y = int(y1+y2)//2 
        cv.circle(frame,(center_x,center_y),4,(255,0,0),-1)
        new_ids.add(id)
 
    num_people = len(new_ids)

    cv.putText(frame, f"People: {num_people}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    

    cv.imshow('FRAME',frame)
    if cv.waitKey(1)&0xFF==27:
        break
cap.release()
cv.destroyAllWindows()