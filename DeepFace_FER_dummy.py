# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:05:12 2022

@author: Ayan
"""

import cv2
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from deepface import DeepFace 
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import os 



face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(r'C:\Users\User\Desktop\haarcascade_frontalface_default.xml')



for file in os.listdir(r"C:\Users\User\Downloads\Test_vid_May17"):
    if file.endswith(".mp4"):
        path=os.path.join(r"C:\Users\User\Downloads\Test_vid_May17", file)
        points=0
        emo_dict={'worry':0,'happy':0,'neutral':0,'sad':0,'surprise':0,'confident':0,'nervous':0}
        cap = cv2.VideoCapture(path)
        
        time=1
        fps = cap.get(cv2.CAP_PROP_FPS) 
        cap.set(5,100)# OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while(1):
            ret,frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (1280, 720))
                labels = []
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,1.3,5)
            else:
            
                break

            #faces = face_cascade.detectMultiScale(frame,scaleFactor = 1.3, minNeighbors = 4 )
            output = DeepFace.analyze(frame, enforce_detection=False, actions=['emotion','dominant_emotion'] )
            #print(output)
            
            emotion=''
            if output['dominant_emotion']=='neutral' and output['emotion']['happy']>20: 
                emotion='confident'
                
            elif output['dominant_emotion']=='sad' and output['emotion']['fear']>20: 
                emotion='nervous'
                
            elif output['dominant_emotion']=='neutral' and output['emotion']['sad']>20: 
                emotion='nervous'
                
            elif output['dominant_emotion']=='neutral' and output['emotion']['fear']>20: 
                emotion='nervous'
    
            elif output['dominant_emotion']=='fear':
                emotion='surprise'
                
            elif output['dominant_emotion']=='happy':
                emotion='happy'
                
            elif output['dominant_emotion']=='sad':
                emotion='sad'
                
            elif output['dominant_emotion']=='neutral':
                emotion='neutral'
                
            elif output['dominant_emotion']=='angry':
                 emotion='worry'
                 
            elif output['dominant_emotion']=='surprise':
                 emotion='surprise'
                 
            elif output['dominant_emotion']=='disgust':
                 emotion='disgust'
                
            
            
            text = str(emotion) 
            
            for (x,y,w,h) in faces :
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)
            
                    cv2.putText(frame,text,(x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), thickness = 2)
                else:
                    cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                
            cv2.imshow("Video", frame)
            if cv2.waitKey(time) & 0xFF == ord('q'):
                break
       
        cap.release()
        cv2.destroyAllWindows()

