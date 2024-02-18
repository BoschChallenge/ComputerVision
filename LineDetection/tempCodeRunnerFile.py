from LineDetection import detectLines
import numpy as np
import cv2
import math

cap = cv2.VideoCapture("full_hd2.mp4")

while True:
    ret,frame = cap.read()
    length,angle,line = detectLines(frame)
    print("length:",length,"angle:",angle,"Right:",line)