import numpy as np
import pyximport; pyximport.install()
from .Functions.TDMAsolver import TDMAsolver
from .Functions.TriDot import TriDot
from scipy import sparse
from scipy import linalg

class Propagator:
    '''Propagate from psi1, in the time t, to psi2, in time t +dt.
       Solves the equation M*psi2 = Mp*psi1 given by the Crank-Nicolson
       Method and returns the propagated Wave Function'''

    def __init__(self, H, N,dt):
        self.H = H.H
        self.N = N
        self.dt = dt
        self.M = [0,0,0]
        self.Mp = [0,0,0]
        self.MatrixSetup()

    def MatrixSetup(self):

        self.M[0] = np.zeros(self.N-1)+1j*self.H[0]*(self.dt)/2
        self.M[1] = np.ones(self.N)+1j*self.H[1]*(self.dt)/2
        self.M[2] = np.zeros(self.N-1)+1j*self.H[2]*(self.dt)/2

        self.Mp[0] = np.zeros(self.N-1)-1j*self.H[0]*(self.dt)/2
        self.Mp[1] = np.ones(self.N)-1j*self.H[1]*(self.dt)/2
        self.Mp[2] = np.zeros(self.N-1)-1j*self.H[2]*(self.dt)/2

    def Propagate(self, psi0):
        self.MatrixSetup()
        #BandedMatrix=np.zeros((3,self.N))
        #BandedMatrix[0]=np.append(0,np.diag(self.M,k=1))
        #BandedMatrix[1]=np.diag(self.M)
        #BandedMatrix[2]=np.append(np.diag(self.M,k=1),0)
        A = self.M
        b = TriDot(self.Mp[0],self.Mp[1],self.Mp[2],psi0)
        #A = sparse.diags([self.M[1],self.M[0],self.M[2]],[0,1,-1]).toarray()
        psi = TDMAsolver(A[0],A[1],A[2],b)
        #psi = linalg.solve(A,b)
        return psi

    def Update(self,H):
        self.H = H.H
