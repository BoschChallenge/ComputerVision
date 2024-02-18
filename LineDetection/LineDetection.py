import numpy as np
import cv2
import math

# cap = cv2.VideoCapture("full_hd2.mp4")

def slope(x1, y1, x2, y2): # Line slope given two points:
    if (x2-x1) != 0:
        return (y2-y1)/(x2-x1)
    else:
        return 0

def angle(s1, s2): 
    return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))

def calculateAngle(lineA,lineB):
    slope1 = slope(lineA[0][0], lineA[0][1], lineA[1][0], lineA[1][1])
    slope2 = slope(lineB[0][0], lineB[0][1], lineB[1][0], lineB[1][1])

    ang = angle(slope1, slope2)
    return ang

def warpImage(img):
    h = img.shape[0]
    w = img.shape[1]
    points = createPoints(w,h,0.2,0.55) #adjust parameters
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))

    return imgWarp
    
#trial and error function
def createPoints(width,height,scalew,scaleh):
    point1 = [scalew*width,scaleh*height]
    point2 = [(1-scalew)*width,scaleh*height]
    point3 = [0,height]
    point4 = [width,height]
    points = [point1,point2,point3,point4]
    
    return points

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
    edges = cv2.Canny(blur,50,150)
    edg = region_of_interest(edges)
    return edg

def region_of_interest(image):
    height = image.shape[0]
    # width = image.shape[1]
    polygons1 = np.array([[(250,330),(50,height),(220,height), (950,330)]])
    # polygons2 = np.array([[(1200,750),(1500,height),(1800,height), (1400,750)]])
    polygons2 = np.array([[(950,330),(1580,height),(1900,height), (1500,330)]])
    # polygons = np.array([[(0,int(height//2)),(0,height),(width,height), (width,int(height//2))]])
    # polygons = np.array([[(125,177),(68,height),(366,height), (305,177)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons1,255)
    cv2.fillPoly(mask,polygons2,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    if x1 < 0 or x1 > 1920 or x2 < 0 or x2 > 1920:
        return np.array([0,0,0,0])
    
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
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 8)

    return line_image

# while True:
#     ret,frame = cap.read()

#     img = warpImage(frame)
#     edges = canny_func(img)
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, np.array([]), minLineLength=40, maxLineGap=5)
#     averaged_lines = average_slope_intercept(img,lines)

#     if averaged_lines is not None:
#         line_image = display_lines(frame,averaged_lines)
#         #Finding distance
#         width = img.shape[1]
#         height = img.shape[0]
#         # MidPoint for distance from line
#         x1 = width // 2
#         y1 = height - int(height//10)
#         x2 = width
#         y2 = y1

#         if len(averaged_lines) != 2:
#             if right_exist:
#                 rx1,ry1,rx2,ry2 = averaged_lines[0].reshape(4)
#                 x,y = line_intersection(([x1,y1],[x2,y2]),([rx1,ry1],[rx2,ry2]))
#                 if(x != 0 and y != 0):
#                     cv2.line(line_image,(x1,y1),(int(x),int(y)),(0,0,255),5)
#                     temp = (x - x1)*(x - x1) + (y - y1)*(y - y1)
#                     length = pow(temp,0.5)
#                     print("length:",length)
#                     angl = calculateAngle(((0,height),(width,height)),((rx1,ry1),(rx2,ry2)))
#                     print("angle:",abs(angl))

#             elif left_exist:
#                 rx1,ry1,rx2,ry2 = averaged_lines[0].reshape(4)
#                 x,y = line_intersection(([x1,y1],[x2,y2]),([rx1,ry1],[rx2,ry2]))
#                 if(x != 0 and y != 0):
#                     cv2.line(line_image,(x1,y1),(int(x),int(y)),(0,0,255),5)
#                     temp = (x - x1)*(x - x1) + (y - y1)*(y - y1)
#                     length = pow(temp,0.5)
#                     print("length:",length)
#                     angl = calculateAngle(((0,height),(width,height)),((rx1,ry1),(rx2,ry2)))
#                     print("angle:",abs(angl))

#         else:
#             rx1,ry1,rx2,ry2 = averaged_lines[1].reshape(4)
#             x,y = line_intersection(([x1,y1],[x2,y2]),([rx1,ry1],[rx2,ry2]))
#             if(x != 0 and y != 0):
#                 cv2.line(line_image,(x1,y1),(int(x),int(y)),(0,0,255),5)
#                 temp = (x - x1)*(x - x1) + (y - y1)*(y - y1)
#                 length = pow(temp,0.5)
#                 print("length:",length)
#                 angl = calculateAngle(((0,height),(width,height)),((rx1,ry1),(rx2,ry2)))
#                 print("angle:",abs(angl))

#         combo_image = cv2.addWeighted(img,0.5,line_image,1,1)
#     else:
#         line_image = np.zeros_like(img)
#         combo_image = cv2.addWeighted(img,0.5,line_image,1,1)

#     cv2.imshow('Result',combo_image)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.relase()
# cv2.destroyAllWindows()

def detectLines(frame):

    img = warpImage(frame)
    edges = canny_func(img)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(img,lines)

    if averaged_lines is not None:
        #Finding distance
        width = img.shape[1]
        height = img.shape[0]
        # MidPoint for distance from line
        x1 = width // 2
        y1 = height - int(height//10)
        x2 = width
        y2 = y1

        if len(averaged_lines) != 2:
            if right_exist:
                rx1,ry1,rx2,ry2 = averaged_lines[0].reshape(4)
                x,y = line_intersection(([x1,y1],[x2,y2]),([rx1,ry1],[rx2,ry2]))
                if(x != 0 and y != 0):
                    temp = (x - x1)*(x - x1) + (y - y1)*(y - y1)
                    length = pow(temp,0.5)
                    angl = calculateAngle(((0,height),(width,height)),((rx1,ry1),(rx2,ry2)))
                    return length,angl,True

            elif left_exist:
                rx1,ry1,rx2,ry2 = averaged_lines[0].reshape(4)
                x,y = line_intersection(([x1,y1],[x2,y2]),([rx1,ry1],[rx2,ry2]))
                if(x != 0 and y != 0):
                    temp = (x - x1)*(x - x1) + (y - y1)*(y - y1)
                    length = pow(temp,0.5)
                    angl = abs(calculateAngle(((0,height),(width,height)),((rx1,ry1),(rx2,ry2))))
                    return length,angl,False
            else:
                return 0,0,False

        else:
            rx1,ry1,rx2,ry2 = averaged_lines[1].reshape(4)
            x,y = line_intersection(([x1,y1],[x2,y2]),([rx1,ry1],[rx2,ry2]))
            if(x != 0 and y != 0):
                temp = (x - x1)*(x - x1) + (y - y1)*(y - y1)
                length = pow(temp,0.5)
                angl = calculateAngle(((0,height),(width,height)),((rx1,ry1),(rx2,ry2)))
                return length,angl,True  

    else:
        return 0,0,False