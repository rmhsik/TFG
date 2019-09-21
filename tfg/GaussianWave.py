import numpy as np

def GWF(a,x0):
    return np.exp(-(x-x0)**2/(2*a**2))
