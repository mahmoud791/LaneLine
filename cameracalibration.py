import cv2
import glob
import numpy as np


def camera_cal():
    x_cor = 9 #Number of corners to find
    y_cor = 6
    objp = np.zeros((y_cor*x_cor,3), np.float32)
    objp[:,:2] = np.mgrid[0:x_cor, 0:y_cor].T.reshape(-1,2)
    
    #Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('camera_cal/calibration*.jpg') # Make a list of paths to calibration images
	# Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        ret, corners = cv2.findChessboardCorners(gray, (x_cor,y_cor), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return mtx, dist