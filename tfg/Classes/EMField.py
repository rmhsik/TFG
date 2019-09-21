import numpy as np
from scipy import integrate

class EMField:
    def __init__(self,Amp,w,t0,tau,tinf=-2.5*15,m=1):
        self.tau = tau
        self.m = m
        self.Amp = Amp
        self.w = w
        self.t0 = t0
        self.tinf = tinf
    def f(self,t):
        #Envelope function from https://doi.org/10.1016/j.physletb.2013.09.014
        return np.exp(-np.power((t-self.t0)/self.tau,2*self.m))

    def E(self,t):
        #Electric field of the laser
        return self.Amp*self.f(t)*np.sin(self.w*t)

    def A(self,t):
        #Vector potential as A(t) = -int(E(t)dt) from t=0 to t
        return -integrate.romberg(self.E, self.tinf, t) #Añadir el tiempo de comienzo
