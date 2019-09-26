import numpy as np
#import potential
from Classes import EMField
c = 137.04

class H:
    def __init__(self,grid,N,h,t,ABool = True,VBool = True):
        self.N = N
        self.H = np.zeros((N,N),dtype="complex")
        self.h = h
        self.x = grid
        self.t = t
        if ABool == True:
            self.A = self.A()
        else:
            self.A = np.zeros(len(self.t))

        if VBool == True:
            self.V = self.V()
        else:
            self.V = np.zeros(len(self.x))
        self.MatrixSetup()

    def V(self):
        V = -1/(np.sqrt(np.power(self.x,2)+1))
        return V

    def A(self):
        EM = EMField.EMField(200,0.63,10,5,-100,5)
        a = np.zeros(len(self.t))
        print("Calculando el potencial vector...")
        for i in range(len(self.t)):
            a[i] = EM.A(self.t[i])
        print("potencial vector calculado!")
        return a #EM.A(t)

    def MatrixSetup(self,j=0):
        #V = 0
        for i in range(self.N):
            try:
                self.H[i,i] = 1/self.h**2 + self.V[i]
                self.H[i,abs(i-1)] = -1/(2*self.h**2)+1j/(2*self.h*c)*self.A[j]
                self.H[i,abs(i+1)] = -1/(2*self.h**2)-1j/(2*self.h*c)*self.A[j]
            except:
                pass

    def BandedMatrix(self):
        V = self.V()
        d = 1/self.h**2 + V
        u = -1/(2*self.h**2)*np.ones(self.N-1)
        l = -1/(2*self.h**2)*np.ones(self.N-1)

        ab = np.zeros((3,self.N))
        ab[0] = np.append([0],u)
        ab[1] = d
        ab[2] = np.append(l,[0])

        return ab

    def Update(self,j):
        self.MatrixSetup(j)
