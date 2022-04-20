from curve_math import *
import numpy as np


EQUID_POINTS = 25 # Number of points to use for the equidistant approximation
IMAGE_H = 223
IMAGE_W = 1280
def equidistant(pol, d, max_l = 1, plot = False):
    y_pol = np.linspace(0, max_l, num=EQUID_POINTS)
    x_pol = pol_calc(pol, y_pol)
    y_pol *= IMAGE_H # Convert y coordinates to [0..223] scale
    x_m = []
    y_m = []
    k_m = []
    for i in range(len(x_pol)-1):
        x_m.append((x_pol[i+1]-x_pol[i])/2.0+x_pol[i]) # Calculate polints position between given points
        y_m.append((y_pol[i+1]-y_pol[i])/2.0+y_pol[i])
        if x_pol[i+1] == x_pol[i]:
            k_m.append(1e8) # A vary big number
        else:
            k_m.append(-(y_pol[i+1]-y_pol[i])/(x_pol[i+1]-x_pol[i])) # Slope of perpendicular lines
    x_m = np.array(x_m)
    y_m = np.array(y_m)
    k_m = np.array(k_m)
  #Calculate equidistant points
    y_eq = d*np.sqrt(1.0/(1+k_m**2))
    x_eq = np.zeros_like(y_eq)
    if d >= 0:
        for i in range(len(x_m)):
            if k_m[i] < 0:
                y_eq[i] = y_m[i]-abs(y_eq[i])
            else:
                y_eq[i] = y_m[i]+abs(y_eq[i])
            x_eq[i] = (x_m[i]-k_m[i]*y_m[i])+k_m[i]*y_eq[i]
    else:
        for i in range(len(x_m)):
            if k_m[i] < 0:
                y_eq[i] = y_m[i]+abs(y_eq[i])
            else:
                y_eq[i] = y_m[i]-abs(y_eq[i])
            x_eq[i] = (x_m[i]-k_m[i]*y_m[i])+k_m[i]*y_eq[i]
    y_eq /= IMAGE_H # Convert all y coordinates back to [0..1] scale
    y_pol /= IMAGE_H
    y_m /= IMAGE_H
    pol_eq = np.polyfit(y_eq, x_eq, len(pol)-1) # Fit equidistant with a polinomial
    return pol_eq