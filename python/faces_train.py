#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:32:36 2019

@author: rat
"""
import os
import cv2
import numpy as np
from PIL import Image
import pickle


y_labels = []
x_train = []
current_id = 0
label_ids = {}

face_casecade = cv2.CascadeClassifier('xml/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            #print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
                print(current_id)
            id_ = label_ids[label]
            
            
            print(label_ids)
            #y_labels.append(label)
            #x_train.append(path)
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image,"uint8") 
            size = (500,500)
            final = pil_image.resize(size,Image.ANTIALIAS)
            #print(image_array)
            faces = face_casecade.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                #print(x_train,id_)
                print(roi)
                x_train.append(roi) 
                y_labels.append(id_)

#print(y_labels)
#print(x_train)
                
with open("labels.pickle","wb") as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")