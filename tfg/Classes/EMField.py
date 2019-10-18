import numpy as np
from scipy import integrate

c = 137.04

class EMField:
    def __init__(self,Amp,w,tmax):
        self.tmax = tmax
        self.Amp = Amp
        self.w = w

    def f(self,t):
        #Envelope function from https://doi.org/10.1016/j.physletb.2013.09.014

        #In case of using the entire t array, search for numpy's fancy indexing
        #for setting all times > tmax to zero "arr[arr>tmax] = 0"
        #f = np.zeros(len(self.t))
        #for i in range(len(self.t)):
        #    if self.t[i]<self.tmax:
        #        f[i] = np.power(np.sin(self.t[i]*np.pi/self.tmax),2)
        #    else:
        #        f[i] = 0.0
        #return f
        #return np.exp(-np.power((t-self.t0)/self.tau,2*self.m))
        if t<=self.tmax:
            return np.power(np.sin(t*np.pi/self.tmax),2)
        else:
            return 0.0
    def E(self,t):
        #Electric field of the laser
        #E = np.zeros(len(t))
        #for i in range(len(t)):
        #    E[i] = self.Amp*self.f(t[i])*np.sin(self.w*t[i])
        #return E
        return self.Amp*self.f(t)*np.sin(self.w*t)
    def A(self,t):
        #Vector potential as A(t) = -int(E(t)dt) from t=0 to t
        #try:
        return -c*integrate.romberg(self.E,0.0,t,divmax=10)
        #except:
            #print("exception encountered!!")
            #pass
