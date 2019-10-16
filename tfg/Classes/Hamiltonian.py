import numpy as np
#import potential
from Classes import EMField
from scipy.linalg.lapack import zheev
import pyximport; pyximport.install()
from .Functions import Diag2Matrix
c = 137.04

class H:
    def __init__(self,x,N,h,t, ABool = True, V = 1, softening = 2.0, R=0.0,
                amp=0.067,w=0.057,tmax=110):
        self.N = N
        self.H = [0,0,0]
        self.h = h
        self.x = x
        self.t = t
        self.soft = softening
        self.amp =amp
        self.w = w
        self.tmax = tmax
        self.R = R
        if ABool == True:
            self.A = self.A()
        else:
            self.A = np.zeros(len(self.t))

        if V == 1:
            self.V = self.V1()
        elif V == 0:
            self.V = np.zeros(len(self.x))
        elif V == 2:
            self.V = self.V2()
        self.MatrixSetup()

    def V1(self):
        V = -1/(np.sqrt(np.power(self.x,2)+self.soft))
        return V

    def V2(self):
        xa = -self.R/2
        xb = self.R/2

        V = -1/(np.sqrt(np.power(self.x-xa,2)+self.soft))-1/(np.sqrt(np.power(self.x-xb,2)+self.soft))
        return V

    def A(self):
        EM = EMField.EMField(self.amp,self.w,self.tmax)
        a = np.zeros(len(self.t))
        print("Calculando el potencial vector...")
        for i in range(len(self.t)):
            a[i] = EM.A(self.t[i])
        print("potencial vector calculado!")
        return a #EM.A(t)

    def MatrixSetup(self,j=0):
        #V = 0
        #for i in range(self.N):
        #    try:
        #        self.H[i,i] = 1/self.h**2 + self.V[i]
        #        self.H[i,abs(i-1)] = -1/(2*self.h**2)+1j/(2*self.h*c)*self.A[j]
        #        self.H[i,abs(i+1)] = -1/(2*self.h**2)-1j/(2*self.h*c)*self.A[j]
        #    except:
        #        pass
        self.H[1] = (1/self.h**2+self.A[j]**2/(2*c**2)+self.V).astype(complex)
        self.H[0] = (-1/(2*self.h**2)+1j/(2*self.h*c)*self.A[j])*np.ones(self.N-1)
        self.H[2] = (-1/(2*self.h**2)-1j/(2*self.h*c)*self.A[j])*np.ones(self.N-1)
        #return [u,d,l]

    def EigenStates(self,s=0):
        self.Matrix = Diag2Matrix.Diag2Matrix(self.H[0],
                                         self.H[1],
                                         self.H[2])
        #self.w,self.v = zheev(Matrix)
    def Update(self,j):
        self.MatrixSetup(j)
