import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def canny_func(image):
    gray = cv2.cvtColor(image,cv2.COLORBGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur,50,150)
    region = region_of_interest(edges)
    return region

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200,height), (500,height), (500,0), (200,0)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame, 1)
    edges = canny_func(frame)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, np.array([]), minLineLength=50, maxLineGap=100)

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
            print('Line found')

    cv2.imshow('Result',frame)
    if cv2.waitKey(1) == ord('q'):
        break


cap.relase()
cv2.destroyAllWindows()