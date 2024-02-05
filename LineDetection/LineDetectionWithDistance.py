import numpy as np
import cv2
from scipy.spatial import distance as dist


cap = cv2.VideoCapture("test2.mp4")


def midpoint(ptA,ptB):
    return ((ptA[0] + ptB[0]) * 0.5,(ptA[1]+ptB[1]) * 0.5)


def canny_func(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur,50,100)
    edg = region_of_interest(edges)
    return edg

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([[(200,height), (1100,height), (550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(2/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
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
            return np.array([right_line])
        elif flagRight:
            left_line = make_coordinates(image,left_fit_average)
            return np.array([left_line])

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
    if not ret:
     print("Error: End of video")
     break
    edges = canny_func(frame)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame,lines)
    if averaged_lines is not None:
        line_image = display_lines(frame,averaged_lines)
    else:
        line_image = display_lines(frame,lines)

    combo_image = cv2.addWeighted(frame,0.8,line_image,1,1)
    
    ####################### DISTANCA ###############################################################

    if averaged_lines is not None:
        line_image = display_lines(frame, averaged_lines)
        
        # Calculate and display distance from left line if there is one, otherwise display distance from right line
        left_line = averaged_lines[0]
        left_midpoint = midpoint((left_line[0], left_line[1]), (left_line[2], left_line[3]))
        left_distance = dist.euclidean((left_midpoint[0], left_midpoint[1]), (frame.shape[1] // 2, frame.shape[0]))
        if len(averaged_lines) != 2:
            cv2.putText(combo_image, f"Right Distance: {left_distance:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(combo_image, (left_line[0] , left_line[1] ), ((frame.shape[1] // 2)  , frame.shape[0] - 50),
			(255, 255, 255), 2)

        
        else:
            right_line = averaged_lines[0]
            right_midpoint = midpoint((right_line[0], right_line[1]), (right_line[2], right_line[3]))
            right_distance = dist.euclidean((right_midpoint[0], right_midpoint[1]), (frame.shape[1] // 2, frame.shape[0]))
            cv2.putText(combo_image, f"Left Distance: {right_distance:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(combo_image, (left_line[0] , left_line[1] ), ((frame.shape[1] // 2)  , frame.shape[0] - 50),
			(255, 255, 255), 2)
            
    else:
        line_image = display_lines(frame, lines)
   

   ###################################################################################################


    cv2.imshow('Result',combo_image)
    if cv2.waitKey(1) == ord('q'):
        break


cap.relase()
cv2.destroyAllWindows()