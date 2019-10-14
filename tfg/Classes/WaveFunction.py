import numpy as np
from scipy import integrate
import pyximport; pyximport.install()
from .Functions import Math
from .Functions.TriDot import TriDot

class WF:
    def __init__(self,a,gridx,gridt,H,x0=0,gamma=1.0):
        self.a = a
        self.x = gridx[0]
        self.h = gridx[1]
        self.t = gridt[0]
        self.dt = gridt[1]
        self.H = H
        self.x0 = x0
        self.gamma = gamma
        self.xb = 10
        self.mask = 0

        self.Mask()
        self.psi=self.WaveFunction()

    def WaveFunction(self):
        psi = np.exp(-(self.x-self.x0)**2/(2*self.a**2))
        C = Math.Norm(psi,self.x)
        return psi.astype(complex)/C


    def Mask(self):
        self.mask = np.ones(len(self.x),dtype='float')
        xb = 0.1*self.x[-1]
        for i in range(len(self.x)):
            if self.x[i]<(self.x[0]+xb):
                self.mask[i] = np.power(np.cos(np.pi*(self.x[i]-(self.x[0]+xb))*self.gamma/(2*xb)),1/8)
            if self.x[i]>(self.x[-1]-xb):
                self.mask[i] = np.power(np.cos(np.pi*(self.x[i]-(self.x[-1]-xb))*self.gamma/(2*xb)),1/8)

    def P(self):
        return np.conjugate(self.psi)*self.psi

    def P2(self):
        return integrate.simps(self.P(),self.x)

    def Energy(self):
        integ = np.conjugate(self.psi)*TriDot(self.H.H[0],self.H.H[1],self.H.H[2],self.psi)
        return integrate.simps(integ,self.x)

    def aExpected(self):
        deltaV = -np.gradient(self.H.V)
        integ = np.conjugate(self.psi)*deltaV*self.psi
        return integrate.simps(integ,self.x)
