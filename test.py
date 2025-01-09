
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model): #ทดสอบรันมีเดียไปป์
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #COLOR BRUH
    image.flags.writeable = False
    results = model.process(image)                 #โค้ดส่วนนี้คือที่ใช้มีเดียไปป์อ่านจอจริง จับจุด
    image.flags.writeable = True
    image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR) #SAME COLOR Ma boi
    return image, results

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

            #อ่าน feed
            ret, frame = cap.read()

            #ตรวจจับจุด
            image, results = mediapipe_detection(frame, holistic)
            print(results)

            cv2.imshow('OpenCV Feed',image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    #สรุปบัค = เว้นระยะบรรทัดผิด
    cap.release()
    cv2.destroyAllWindows()

len(results.face_landmarks.landmark)




