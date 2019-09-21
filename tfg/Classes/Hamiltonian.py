import numpy as np
#import potential

class H:
    def __init__(self,grid,N,h):
        self.N = N
        self.H = np.zeros((N,N))
        self.h = h
        self.x = grid
        self.MatrixSetup()

    def V(self):
        V = -1/(np.sqrt(np.power(self.x,2)+1))
        return V

    def MatrixSetup(self):
        V = self.V()
        for i in range(self.N):
            try:
                self.H[i,i] = 1/self.h**2 + V[i]
                if i-1 >= 0:
                    self.H[i,i-1] = -1/(2*self.h**2)
                if i+1 <= self.N:
                    self.H[i,i+1] = -1/(2*self.h**2)
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
