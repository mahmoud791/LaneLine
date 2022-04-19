
# Load nessesery modules and set up
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from curve_math import *
from thresholding import s_hls
from Equidistant import equidistant
from cameracalibration import camera_cal
from sklearn.metrics import mean_squared_error


IMAGE_H = 223
IMAGE_W = 1280
EQUID_POINTS = 25 # Number of points to use for the equidistant approximation
WINDOW_SIZE = 15 # Half of the sensor span
DEV = 7 # Maximum of the point deviation from the sensor center
SPEED = 2 / IMAGE_H # Pixels shift per frame
POL_ORD = 2 # Default polinomial order
RANGE = 0.0 # Fraction of the image to skip
DEV_POL = 2 # Max mean squared error of the approximation
MSE_DEV = 1.1 # Minimum mean squared error ratio to consider higher order of the polynomial
RANGE = 0.0

right_fit_p = np.zeros(POL_ORD+1)
left_fit_p = np.zeros(POL_ORD+1)
r_len = 0
l_len = 0
lane_w_p = 90

MIN = 60 # Minimal line separation (in px)
MAX = 95 # Maximal line separation (in px)
MIN_POINTS = 10  #Minimal points to consider a line
MAX_N = 5 # Maximal frames without line detected to use previous frame
n_count = 0 # Frame counter
r_n = 0 # Number of frames with unsuccessful line detection
l_n = 0

mtx = []
dist = []
M = []
Minv = []




def undistort(img):
	return cv2.undistort(img, mtx, dist, None, mtx)

# Sharpen image
def sharpen_img(img):
    gb = cv2.GaussianBlur(img, (5,5), 20.0)
    return cv2.addWeighted(img, 2, gb, -1, 0)

# Compute linear image transformation img*s+m
def lin_img(img,s=1.0,m=0.0):
    img2=cv2.multiply(img, np.array([s]))
    return cv2.add(img2, np.array([m]))

# Change image contrast; s>1 - increase
def contr_img(img, s=1.0):
    m=127.0*(1.0-s)
    return lin_img(img, s, m)


# Create perspective image transformation matrices
def create_M():
    src = np.float32([[0, 673], [1207, 673], [0, 450], [1280, 450]])
    dst = np.float32([[569, 223], [711, 223], [0, 0], [1280, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

# Main image transformation routine to get a warped image
def transform(img, M):
    undist = undistort(img)
    img_size = (IMAGE_W, IMAGE_H)
    warped = cv2.warpPerspective(undist, M, img_size)
    warped = sharpen_img(warped)
    warped = contr_img(warped, 1.1)
    return warped







def draw_lane(img, video=False): #Draw found lane line onto a normal image
    if video:
        img, left_fitx, right_fitx, ploty, left, right = get_lane_video(img)
    else:
        img, left_fitx, right_fitx, ploty = get_lane(img, True)
    warp_zero = np.zeros((IMAGE_H,IMAGE_W)).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1.0, newwarp, 0.6, 0)
    if video:
        # Add text information on the video frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_pos = 'Pos of the car: '+str(np.round(lane_offset(left, right),2))+ ' m'
        radius = np.round(lane_curv(left, right),2)
        if radius >= MAX_RADIUS:
            radius = 'Inf'
        else:
            radius = str(radius)
        text_rad = 'Radius: '+radius+ ' m'
        cv2.putText(result,text_pos,(10,25), font, 1,(255,255,255),2)
        cv2.putText(result,text_rad,(10,75), font, 1,(255,255,255),2)
        return(result)









def init_params(ran):
    global right_fit_p, left_fit_p, n_count, RANGE, MIN_POINTS
    right_fit_p = np.zeros(POL_ORD+1)
    left_fit_p = np.zeros(POL_ORD+1)
    n_count = 0
    RANGE = ran
    MIN_POINTS = 25-15*ran
    
def process_image(image):
    return draw_lane(image, True)


# Process videos

def main():
    init_params(0.2)
    global mtx , dist,M,Minv
    mtx, dist = camera_cal() # camera calibration matrix And distortion matrix
    M, Minv = create_M() # bird eye transformation matrix and inverse matrix
    output_v = 'project_video_proc.mp4'
    clip1 = VideoFileClip("challenge_video.mp4")
    clip = clip1.fl_image(process_image)
    clip.write_videofile(output_v, audio=False)


if __name__ == '__main__':
    main()