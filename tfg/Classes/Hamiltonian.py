import numpy as np
#import potential
c = 137.04

class H:
    def __init__(self,grid,N,h):
        self.N = N
        self.H = np.zeros((N,N),dtype="complex")
        self.h = h
        self.x = grid

        self.MatrixSetup()

    def V(self):
        V = -1/(np.sqrt(np.power(self.x,2)+1))
        return V

    def A(self,t):
        return 1000

    def MatrixSetup(self,t=0):
        V = self.V()
        #V = 0
        for i in range(self.N):
            try:
                self.H[i,i] = 1/self.h**2 #+ V[i]
                self.H[i,abs(i-1)] = -1/(2*self.h**2)+1j/(2*self.h*c)*self.A(t)
                self.H[i,abs(i+1)] = -1/(2*self.h**2)-1j/(2*self.h*c)*self.A(t)
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

    def Update(self,t):
        self.MatrixSetup(t)
