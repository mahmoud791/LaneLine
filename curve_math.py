
import numpy as np
from sklearn.metrics import mean_squared_error


IMAGE_H = 223
IMAGE_W = 1280
EQUID_POINTS = 25


#Calculate coefficients of a polynomial in y+h coordinates, i.e. f(y) -> f(y+h)
def pol_shift(pol, h):
    pol_ord = len(pol)-1 # Determinate degree of the polynomial 
    if pol_ord == 3:
        pol0 = pol[0]
        pol1 = pol[1] + 3.0*pol[0]*h
        pol2 = pol[2] + 3.0*pol[0]*h*h + 2.0*pol[1]*h
        pol3 = pol[3] + pol[0]*h*h*h + pol[1]*h*h + pol[2]*h
        return(np.array([pol0, pol1, pol2, pol3]))
    if pol_ord == 2:
        pol0 = pol[0]
        pol1 = pol[1] + 2.0*pol[0]*h
        pol2 = pol[2] + pol[0]*h*h+pol[1]*h
        return(np.array([pol0, pol1, pol2]))
    if pol_ord == 1:
        pol0 = pol[0]
        pol1 = pol[1] + pol[0]*h
        return(np.array([pol0, pol1]))


# Calculate derivative for a polynomial pol in a point x
def pol_d(pol, x):
    pol_ord = len(pol)-1
    if pol_ord == 3:
        return 3.0*pol[0]*x*x+2.0*pol[1]*x+pol[2]
    if pol_ord == 2:
        return 2.0*pol[0]*x+pol[1]
    if pol_ord == 1:
        return pol[0]#*np.ones(len(np.array(x)))
    
# Calculate the second derivative for a polynomial pol in a point x
def pol_dd(pol, x):
    pol_ord = len(pol)-1
    if pol_ord == 3:
        return 6.0*pol[0]*x+2.0*pol[1]
    if pol_ord == 2:
        return 2.0*pol[0]
    if pol_ord == 1:
        return 0.0
    
# Calculate a polinomial value in a given point x
def pol_calc(pol, x):
    pol_f = np.poly1d(pol)
    return(pol_f(x))

xm_in_px = 3.675 / 85 # Lane width (12 ft in m) is ~85 px on image
ym_in_px = 3.048 / 24 # Dashed line length (10 ft in m) is ~24 px on image

def px_to_m(px): # Conver ofset in pixels in x axis into m
    return xm_in_px*px

# Calculate offset from the lane center
def lane_offset(left, right):
    offset = IMAGE_W/2.0-(pol_calc(left, 1.0)+ pol_calc(right, 1.0))/2.0
    return px_to_m(offset)

# Calculate radius of curvature of a line
MAX_RADIUS = 10000
def r_curv(pol, y):
    if len(pol) == 2: # If the polinomial is a linear function
        return MAX_RADIUS
    else:
        y_pol = np.linspace(0, 1, num=EQUID_POINTS)
        x_pol = pol_calc(pol, y_pol)*xm_in_px
        y_pol = y_pol*IMAGE_H*ym_in_px
        pol = np.polyfit(y_pol, x_pol, len(pol)-1)
        d_y = pol_d(pol, y)
        dd_y = pol_dd(pol, y)
        r = ((np.sqrt(1+d_y**2))**3)/abs(dd_y)
        if r > MAX_RADIUS:
            r = MAX_RADIUS
        return r
# Calculate radius of curvature of a lane by avaraging lines curvatures
def lane_curv(left, right):
    l = r_curv(left, 1.0)
    r = r_curv(right, 1.0)
    if l < MAX_RADIUS and r < MAX_RADIUS:
        return (r_curv(left, 1.0)+r_curv(right, 1.0))/2.0
    else:
        if l < MAX_RADIUS:
            return l
        if r < MAX_RADIUS:
            return r
        return MAX_RADIUS


DEV_POL = 2 # Max mean squared error of the approximation
MSE_DEV = 1.1 # Minimum mean squared error ratio to consider higher order of the polynomial
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


# Calculate threashold for left line
def thres_l_calc(sens):
    thres = -0.0045*sens**2+1.7581*sens-115.0
    if thres < 25*(382.0-sens)/382.0+5:
        thres = 25*(382.0-sens)/382.0+5
    return thres

# Calculate threashold for right line
def thres_r_calc(sens):
    thres = -0.0411*sens**2+9.1708*sens-430.0
    if sens<210:
        if thres < sens/6:
            thres = sens/6
    else:
        if thres < 20:
            thres = 20
    return thres


