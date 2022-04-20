
# Load nessesery modules and set up
from fileinput import filename
import sys
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


# Choose the best polynomial order to fit points (x,y)

def best_pol_ord(x, y):
    pol1 = np.polyfit(y,x,1)
    pred1 = pol_calc(pol1, y)
    mse1 = mean_squared_error(x, pred1)
    if mse1 < DEV_POL:
        return pol1, mse1
    pol2 = np.polyfit(y,x,2)
    pred2 = pol_calc(pol2, y)
    mse2 = mean_squared_error(x, pred2)
    if mse2 < DEV_POL or mse1/mse2 < MSE_DEV:
            return pol2, mse2
    else:
        pol3 = np.polyfit(y,x,3)
        pred3 = pol_calc(pol3, y)
        mse3 = mean_squared_error(x, pred3)
        if mse2/mse3 < MSE_DEV:
            return pol2, mse2
        else:
            return pol3, mse3
    
# Smooth polinomial functions of different degrees   
def smooth_dif_ord(pol_p, x, y, new_ord):
    x_p = pol_calc(pol_p, y)
    x_new = (x+x_p)/2.0
    return np.polyfit(y, x_new, new_ord)


# Create virtual sensors
def find(img, left=True, p_ord=POL_ORD, pol = np.zeros(POL_ORD+1), max_n = 0):
    x_pos = [] #lists of found points
    y_pos = []
    max_l = img.shape[0] #number of lines in the img
    for i in range(max_l-int(max_l*RANGE)):
        y = max_l-i #Line number
        y_01 = y / float(max_l) #y in [0..1] scale
        if abs(pol[-1]) > 0: #If it not a still image or the first video frame
            if y_01 >= max_n + SPEED: # If we can use pol to find center of the virtual sensor from the previous frame 
                cent = int(pol_calc(pol, y_01-SPEED))
                if y == max_l:
                    if left:
                        cent = 605
                    else:
                        cent = 690
            else: # Prolong the pol tangentially
                k = pol_d(pol, max_n)
                b = pol_calc(pol, max_n)-k*max_n
                cent = int(k*y_01+b)
            if cent > IMAGE_W-WINDOW_SIZE:
                cent = IMAGE_W-WINDOW_SIZE
            if cent < WINDOW_SIZE:
                cent = WINDOW_SIZE
        else: #If it is a still image
            if len(x_pos) > 0: # If there are some points detected
                cent = x_pos[-1] # Use the previous point as a senser center
            else: #Initial guess on line position
                if left:
                    cent = 605
                else:
                    cent = 690
        if left: #Subsample image
            sens = 0.5*s_hls(img[max_l-1-i:max_l-i,cent-WINDOW_SIZE:cent+WINDOW_SIZE,:])            +img[max_l-1-i:max_l-i,cent-WINDOW_SIZE:cent+WINDOW_SIZE,2]
        else:
            sens = img[max_l-1-i:max_l-i,cent-WINDOW_SIZE:cent+WINDOW_SIZE,2] # Red channel only for right white line
        if len(sens[0,:]) < WINDOW_SIZE: #If we out of the image
                break
        x_max = max(sens[0,:]) #Find maximal value on the sensor
        sens_mean = np.mean(sens[0,:])
        # Get threshold
        if left:
            loc_thres = thres_l_calc(sens_mean)
            loc_dev = DEV
        else:
            loc_thres = thres_r_calc(sens_mean)
            loc_dev = DEV
        if len(x_pos) == 0:
            loc_dev = WINDOW_SIZE
        if (x_max-sens_mean) > loc_thres and (x_max>100 or left):
            if left:
                x = list(reversed(sens[0,:])).index(x_max)
                x = cent+WINDOW_SIZE-x
            else:
                x = list(sens[0,:]).index(x_max)
                x = cent-WINDOW_SIZE+x
            if x-1 < 569.0*y_01 or x+1 > 569.0*y_01+711 : #if the sensor touchs black triangle
                break # We are done
            if abs(pol[-1]) < 1e-4:  # If there are no polynomial provided     
                x_pos.append(x)
                y_pos.append(y_01)
            else:
                if abs(x-cent) < loc_dev: # If the found point deviated from expected position not significantly
                    x_pos.append(x)
                    y_pos.append(y_01)
    if len(x_pos) > 1:
        return x_pos, y_pos
    else:
        return [0], [0.0]




def find_img(img, left=True, p_ord=POL_ORD, pol = np.zeros(POL_ORD+1), max_n = 0):
    x_pos = [] #lists of found points
    y_pos = []
    max_l = img.shape[0] #number of lines in the img
    b_img = np.zeros_like(img)
    b_img = b_img[:,:,2]
    for i in range(max_l-int(max_l*RANGE)):
        y = max_l-i #Line number
        y_01 = y / float(max_l) #y in [0..1] scale
        if len(x_pos) > 0: # If there are some points detected
            cent = x_pos[-1] # Use the previous point as a senser center
        else: #Initial guess on line position
            if left:
                cent = 605
            else:
                cent = 690
        if left: #Subsample image
            sens = 0.5*s_hls(img[max_l-1-i:max_l-i,cent-WINDOW_SIZE:cent+WINDOW_SIZE,:])\
            +img[max_l-1-i:max_l-i,cent-WINDOW_SIZE:cent+WINDOW_SIZE,2]
        else:
            sens = img[max_l-1-i:max_l-i,cent-WINDOW_SIZE:cent+WINDOW_SIZE,2] # Red channel only for right white line
        if len(sens[0,:]) < WINDOW_SIZE: #If we out of the image
                break
        x_max = max(sens[0,:]) #Find maximal value on the sensor
        sens_mean = np.mean(sens[0,:])
        # Get threshold
        if left:
            loc_thres = thres_l_calc(sens_mean)
            loc_dev = DEV
        else:
            loc_thres = thres_r_calc(sens_mean)
            loc_dev = DEV
        #print((sens-sens_mean)>loc_thres)
        b_img[max_l-1-i:max_l-i,cent-WINDOW_SIZE:cent+WINDOW_SIZE]=(sens-sens_mean)>loc_thres
        if len(x_pos) == 0:
            loc_dev = WINDOW_SIZE
        if (x_max-sens_mean) > loc_thres and (x_max>100 or left):
            if left:
                x = list(reversed(sens[0,:])).index(x_max)
                x = cent+WINDOW_SIZE-x
            else:
                x = list(sens[0,:]).index(x_max)
                x = cent-WINDOW_SIZE+x
            if x-1 < 569.0*y_01 or x+1 > 569.0*y_01+711 : #if the sensor touchs black triangle
                break # We are done
            if abs(pol[-1]) < 1e-4:  # If there are no polynomial provided     
                x_pos.append(x)
                y_pos.append(y_01)
            else:
                if abs(x-cent) < loc_dev: # If the found point deviated from expected position not significantly
                    x_pos.append(x)
                    y_pos.append(y_01)
    
    img_proc=np.uint8(255*b_img/np.max(b_img))
    #show_img(img_proc)
    if len(x_pos) > 1:
        return x_pos, y_pos, img_proc
    else:
        return [0], [0.0],img_proc


# Get lines on a still image


def get_lane_video(img):
    global right_fit_p, left_fit_p, r_len, l_len, n_count, r_n, l_n
    sw = False
    warp = transform(img, M)
    img = undistort(img)


    
    x2, y2, im1 = find_img(warp.copy())
    x, y, im2 = find_img(warp.copy(), False)

    considered_points = np.zeros((im1.shape[0],im1.shape[1],3))
    considered_points[:,:,0] = im1+im2
    considered_points[:,:,1] = im1+im2
    considered_points[:,:,2] = im1+im2
    
    

    if l_n < MAX_N and n_count > 0:
        x, y = find(warp, pol = left_fit_p, max_n = l_len)
    else:
        x, y = find(warp)
    if len(x) > MIN_POINTS:
        left_fit, mse_l = best_pol_ord(x,y)
        if mse_l > DEV_POL*9 and n_count > 0:
            left_fit = left_fit_p
            l_n += 1
        else:
            l_n /= 2
    else:
        left_fit = left_fit_p
        l_n += 1
    if r_n < MAX_N and n_count > 0:
        x2, y2 = find(warp, False, pol = right_fit_p, max_n = r_len)
    else:
        x2, y2 = find(warp, False)
    if len(x2) > MIN_POINTS:
        right_fit, mse_r = best_pol_ord(x2, y2)
        if mse_r > DEV_POL*9 and n_count > 0:
            right_fit = right_fit_p
            r_n += 1
        else:
            r_n /= 2
    else:
        right_fit = right_fit_p
        r_n += 1
    if n_count > 0: # if not the first video frame
        # Apply filter
        if len(left_fit_p) == len(left_fit): # If new and prev polinomial have the same order
            left_fit = pol_shift(left_fit_p, -SPEED)*(1.0-len(x)/((1.0-RANGE)*IMAGE_H))+left_fit*(len(x)/((1.0-RANGE)*IMAGE_H))
        else:
            left_fit = smooth_dif_ord(left_fit_p, x, y, len(left_fit)-1)
        l_len = y[-1]
        if len(right_fit_p) == len(right_fit):
            right_fit = pol_shift(right_fit_p, -SPEED)*(1.0-len(x2)/((1.0-RANGE)*IMAGE_H))+right_fit*(len(x2)/((1.0-RANGE)*IMAGE_H))
        else:
            right_fit = smooth_dif_ord(right_fit_p, x2, y2, len(right_fit)-1)
        r_len = y2[-1]
        
    if len(x) > MIN_POINTS and len(x2) <= MIN_POINTS: # If we have only left line
        lane_w = pol_calc(right_fit_p, 1.0)-pol_calc(left_fit_p, 1.0)
        right_fit = smooth_dif_ord(right_fit_p, pol_calc(equidistant(left_fit, lane_w, max_l=l_len), y),
                                   y, len(left_fit)-1)
        r_len = l_len
        r_n /=2
    if len(x2) > MIN_POINTS and len(x) <= MIN_POINTS: # If we have only right line
        lane_w = pol_calc(right_fit_p, 1.0)-pol_calc(left_fit_p, 1.0)
        #print(lane_w)
        left_fit = smooth_dif_ord(left_fit_p, pol_calc(equidistant(right_fit, -lane_w, max_l=r_len), y2),
                                  y2, len(right_fit)-1)
        l_len = r_len
        l_n /=2 
    if (l_n < MAX_N and r_n < MAX_N):
        max_y = max(RANGE, l_len, r_len)
    else:
        max_y = 1.0#max(RANGE, l_len, r_len)
        sw = True
    d1 = pol_calc(right_fit, 1.0)-pol_calc(left_fit, 1.0)
    dm = pol_calc(right_fit, max_y)-pol_calc(left_fit, max_y)
    if (d1 > MAX or d1 < 60 or dm < 0):
        left_fit = left_fit_p
        right_fit = right_fit_p
        l_n += 1
        r_n += 1
    ploty = np.linspace(max_y, 1, num=IMAGE_H) 
    left_fitx = pol_calc(left_fit, ploty)
    right_fitx = pol_calc(right_fit, ploty)
    right_fit_p = np.copy(right_fit)
    left_fit_p = np.copy(left_fit)
    n_count += 1
    return img,  left_fitx, right_fitx, ploty*223.0, left_fit, right_fit,warp,considered_points

def get_lane(img, plot=False):
    warp = transform(img, M)
    img = undistort(img)
    ploty = np.linspace(0, 1, num=warp.shape[0])
    x2, y2 = find(warp)
    x, y = find(warp, False)
    right_fitx = pol_calc(best_pol_ord(x,y)[0], ploty)
    left_fitx = pol_calc(best_pol_ord(x2,y2)[0], ploty)
    y2 = np.int16(np.array(y2)*223.0) # Convert into [0..223] scale
    y = np.int16(np.array(y)*223.0)
    return img,  left_fitx, right_fitx, ploty*IMAGE_H
            
def draw_lane_img_p(img_path): # Image read function
    return cv2.imread(img_path)

def draw_lane(img, video=False): #Draw found lane line onto a normal image
    if video:
        img, left_fitx, right_fitx, ploty, left, right,warp,considered_points = get_lane_video(img)
        
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
        return(result,warp,considered_points,color_warp,newwarp)
   
# Function to initialize parameters before each video processing
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

    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    debug = sys.argv[3]


    init_params(0.2)
    global mtx , dist,M,Minv
    mtx, dist = camera_cal() # camera calibration matrix And distortion matrix
    M, Minv = create_M() # bird eye transformation matrix and inverse matrix
    output_4 = output_path + '_final.mp4'
    output_3 = output_path + '_3.mp4'
    output_2 = output_path + '_2.mp4'
    output_1 = output_path + '_1.mp4'

    clip1 = VideoFileClip(input_path)

    images = [i for i in clip1.iter_frames()]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    size = images[0].shape[1], images[0].shape[0]
    writer_4 = cv2.VideoWriter(output_4, fourcc, 30, size)
    

    if debug == 'y' or debug == 'Y':
        writer_3 = cv2.VideoWriter(output_3, fourcc, 30, size)
        writer_2 = cv2.VideoWriter(output_2, fourcc, 30, (1280,223))
        writer_1 = cv2.VideoWriter(output_1, fourcc, 30, (1280,223))
        for frame in images:
            frame,warp,considered_points,color_warp,newwarp = process_image(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #print(considered_points.shape)
            cv2.imshow("FINAL OUTPUT", frame)
            writer_4.write(frame)
            writer_3.write(newwarp)
            writer_2.write(color_warp)
            writer_1.write(warp)
            # Display frame for X milliseconds and check if q key is pressed
            # q == quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break

    else:
        for frame in images:
            frame,warp,considered_points,color_warp,newwarp = process_image(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("FINAL OUTPUT", frame)
            writer_4.write(frame)
            # Display frame for X milliseconds and check if q key is pressed
            # q == quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
    



    #clip = clip1.fl_image(process_image)
    #clip.write_videofile(output_v, audio=False)



if __name__ == '__main__':
    main()

