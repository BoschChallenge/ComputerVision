import numpy as np
import cv2 as cv

class ImageTransform:
    def __init__(self) -> None:
        pass

    def rescaleFrame(self,image,scale=0.20):
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dimensions = (width,height)

        return cv.resize(image,dimensions,interpolation=cv.INTER_AREA)

    def transform(self,img):
        w = img.shape[1]
        h = img.shape[0]
        points = self.createPoints(w,h,0.2,0.7) #adjust parameters
        pts1 = np.float32(points)
        pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        matrix = cv.getPerspectiveTransform(pts1,pts2)
        imgWarp = cv.warpPerspective(img,matrix,(w,h))

        return imgWarp
        
    #trial and error function
    def createPoints(self,width,height,scalew,scaleh):
        point1 = [scalew*width,scaleh*height]
        point2 = [(1-scalew)*width,scaleh*height]
        point3 = [0,height]
        point4 = [width,height]
        points = [point1,point2,point3,point4]
        
        return points