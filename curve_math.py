
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
