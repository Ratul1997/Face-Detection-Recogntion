import cv2
import pickle
import numpy as np
import os

face_casecade = cv2.CascadeClassifier('xml/haarcascade_frontalface_alt2.xml')
eye_casecade = cv2.CascadeClassifier('xml/haarcascade_eye.xml')
smile_casecade = cv2.CascadeClassifier('xml/haarcascade_smile.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name":1}
with open("labels.pickle", "rb") as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}
        
if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    img_counter = 0

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_casecade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
        
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            id_,conf = recognizer.predict(roi_gray)
            #print(id_)
            if conf>=45 :
                print(id_)
                print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255,255,255)
                stroke = 2
                cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        
            img_item = "my-image.png"
            cv2.imwrite(img_item, gray)
            color = (255,0,0)
            stroke = 2
            end_cord_x = x+w
            end_cord_y = h+y
            cv2.rectangle(frame,(x,y),(end_cord_x, end_cord_y),color,stroke)
           # eyes = smile_casecade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in eyes:
             #   cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
        cv2.imshow("test", frame)
        
        
        if not ret:
            break
        k = cv2.waitKey(30)
         
        
        
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            
    
    
    cam.release()
    cv2.destroyAllWindows()