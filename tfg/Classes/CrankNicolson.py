import numpy as np
from scipy import sparse
from scipy.sparse import linalg

class Propagator:
    '''Propagate from psi1, in the time t, to psi2, in time t +dt.
       Solves the equation M*psi2 = Mp*psi1 given by the Crank-Nicolson
       Method and returns the propagated Wave Function'''

    def __init__(self, H, N,dt):
        self.H = H.H
        self.N = N
        self.dt = dt

        self.MatrixSetup()

    def MatrixSetup(self,H=0):
        if type(H) == int:
            H = self.H
        self.M = np.identity(self.N)+1j*H*(self.dt)/2
        self.Mp = np.identity(self.N)-1j*H*(self.dt)/2

    def Propagate(self, psi0):
        self.MatrixSetup()
        #BandedMatrix=np.zeros((3,self.N))
        #BandedMatrix[0]=np.append(0,np.diag(self.M,k=1))
        #BandedMatrix[1]=np.diag(self.M)
        #BandedMatrix[2]=np.append(np.diag(self.M,k=1),0)
        A = sparse.csc_matrix(self.M)
        b =np.dot(self.Mp,psi0)
        psi = linalg.spsolve(A,b)
        return psi

    def Update(self,H):
        self.H = H.H
