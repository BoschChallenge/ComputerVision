import pickle
import cv2

img = cv2.imread("img13.png")

with open('calibration.pkl', 'rb') as file:
    # Ucitavanje parametara iz calibration.pkl
    loaded_data = pickle.load(file)
    cameraMatrix,dist = loaded_data
    # Proces kalibarcije
    h = img.shape[0]
    w = img.shape[1]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

    # Undistort with Remapping
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    calibratedImage = dst[y:y+h, x:x+w]
    
    cv2.imshow("Result",calibratedImage)
    cv2.waitKey(0)