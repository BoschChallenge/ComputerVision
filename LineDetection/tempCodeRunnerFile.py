from LineDetection import LineDetect
import numpy as np
import cv2
import math

cap = cv2.VideoCapture("full_hd2.mp4")
LD = LineDetect()

while True:
    ret,frame = cap.read()
    length,angle,line = LD.detectLines(frame)
    print("length:",length,"angle:",angle,"Right:",line)