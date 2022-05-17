# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:05:12 2022

@author: Ayan
"""

import cv2
import matplotlib.pyplot as plt 
from deepface import DeepFace 
import os 



face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(r'C:\Users\User\Desktop\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(r"C:\Users\User\Downloads\Video_test_folder\Best_vid.mp4")


for file in os.listdir(r"C:\Users\User\Downloads\Video_test_folder"):
    if file.endswith(".mp4"):
        path=os.path.join(r"C:\Users\User\Downloads\Video_test_folder", file)
        cap = cv2.VideoCapture(path)
        
        while(1):
            ret,frame = cap.read()
            if not ret :
                print('error reading frames..')
                break 
            faces = face_cascade.detectMultiScale(frame,scaleFactor = 1.3, minNeighbors = 4 )
            output = DeepFace.analyze(frame) 
            
            emotion=''
            if output['dominant_emotion']=='neutral' and output['emotion']['happy']>20: 
                emotion='confident'
            elif output['dominant_emotion']=='sad' and output['emotion']['fear']>20: 
                emotion='nervous'
            elif output['dominant_emotion']=='fear':
                emotion='surprise'
            else:
                emotion=output['dominant_emotion']
            print(emotion)
            text = str(emotion) 
            
            for (x,y,w,h) in faces :
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2) 
                cv2.putText(frame,text,(x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), thickness = 2)
                
            cv2.imshow("Video", frame)
            
            if cv2.waitKey(1) &0xFF==ord('a'):
                break
        cap.release()
        cv2.destroyAllWindows()