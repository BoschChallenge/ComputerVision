import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("test_4.mp4")

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return 0,0

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def canny_func(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur,50,100)
    edg = region_of_interest(edges)
    return edg

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([[(0,int(height//2)),(0,height),(width,height), (width,int(height//2))]])
    # polygons = np.array([[(125,177),(68,height),(366,height), (305,177)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3.7/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    global left_exist
    global right_exist 
    left_exist = False
    right_exist = False
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4) #IDEA if x1 is less than half of the width it is left line and same for opposite
            parameters = np.polyfit((x1,x2),(y1,y2),1)  
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope,intercept))
            else:
                right_fit.append((slope,intercept))

        if len(left_fit) != 0:
            left_fit_average = np.average(left_fit, axis = 0)
        else:
            left_fit_average = np.array(np.NaN)
        if len(right_fit) != 0:
            right_fit_average = np.average(right_fit, axis = 0)
        else:
            right_fit_average = np.array(np.NaN)

        size1 = np.size(left_fit_average)
        size2 = np.size(right_fit_average)
        flagLeft = False
        flagRight = False

        if size1 == 1 and np.isnan(left_fit_average) == True:
                flagLeft = True
        if size2 == 1 and np.isnan(right_fit_average) == True:
                flagRight = True

        if flagLeft and flagRight:
            return None
        elif flagLeft:
            right_line = make_coordinates(image,right_fit_average)
            right_exist = True
            return np.array([right_line])
        elif flagRight:
            left_line = make_coordinates(image,left_fit_average)
            left_exist = True
            return np.array([left_line])

        left_exist = True
        right_exist = True
        left_line = make_coordinates(image,left_fit_average)
        right_line = make_coordinates(image,right_fit_average)

        return np.array([left_line,right_line])
    return None

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    return line_image

while True:
    ret,frame = cap.read()
    
    edges = canny_func(frame)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame,lines)

    if averaged_lines is not None:
        line_image = display_lines(frame,averaged_lines)
        #Finding distance
        width = frame.shape[1]
        height = frame.shape[0]
        print("width: ",width,"height: ",height)
        if len(averaged_lines) != 2:
            if right_exist:
                x1 = width // 2
                y1 = height - int(height//10)
                x2 = width
                y2 = y1
                rx1,ry1,rx2,ry2 = averaged_lines[0].reshape(4)
                x,y = line_intersection(([x1,y1],[x2,y2]),([rx1,ry1],[rx2,ry2]))
                if(x != 0 and y != 0):
                    cv2.line(line_image,(x1,y1),(int(x),int(y)),(0,0,255),5)
                    temp = (x - x1)*(x - x1) + (y - y1)*(y - y1)
                    length = pow(temp,0.5)
                    print(length)
            elif left_exist:
                x1 = width // 2
                y1 = height - int(height//10)
                x2 = 0
                y2 = y1
                rx1,ry1,rx2,ry2 = averaged_lines[0].reshape(4)
                x,y = line_intersection(([x1,y1],[x2,y2]),([rx1,ry1],[rx2,ry2]))
                if(x != 0 and y != 0):
                    cv2.line(line_image,(x1,y1),(int(x),int(y)),(0,0,255),5)
                    temp = (x - x1)*(x - x1) + (y - y1)*(y - y1)
                    length = pow(temp,0.5)
                    print(length)
        else:
            x1 = width // 2
            y1 = height - int(height//10)
            x2 = width
            y2 = y1
            rx1,ry1,rx2,ry2 = averaged_lines[1].reshape(4)
            x,y = line_intersection(([x1,y1],[x2,y2]),([rx1,ry1],[rx2,ry2]))
            if(x != 0 and y != 0):
                cv2.line(line_image,(x1,y1),(int(x),int(y)),(0,0,255),5)
                temp = (x - x1)*(x - x1) + (y - y1)*(y - y1)
                length = pow(temp,0.5)
                print(length)
    else:
        line_image = display_lines(frame,lines)

    
    combo_image = cv2.addWeighted(frame,0.8,line_image,1,1)

    cv2.imshow('Result',combo_image)
    if cv2.waitKey(1) == ord('q'):
        break


cap.relase()
cv2.destroyAllWindows()