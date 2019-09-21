import numpy as np
from scipy import linalg

class Propagator:
    '''Propagate from psi1, in the time t, to psi2, in time t +dt.
       Solves the equation M*psi2 = Mp*psi1 given by the Crank-Nicolson
       Method and returns the propagated Wave Function'''

    def __init__(self, H, dt, N):
        self.H = H.H
        self.dt = dt
        self.N = N

    def MatrixSetup(self):
        self.M = np.identity(self.N)+1j*self.H*(self.dt)/2
        self.Mp = np.identity(self.N)-1j*self.H*(self.dt)/2

    def Propagate(self, psi0):
        self.MatrixSetup()
        psi = linalg.spsolve(self.M,np.dot(self.Mp,psi0))
        return psi

#A TEST CHANGE
