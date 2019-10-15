import numpy as np
from .CrankNicolson import Propagator
from .Hamiltonian import H as Ham
from .Functions.GroundState import GroundState
from scipy import integrate
import pyximport; pyximport.install()
from .Functions import Math
from .Functions.TriDot import TriDot

class WF:
    def __init__(self,dict_params):
        self.dict = dict_params
        self.Nx         = self.dict["Nx"]
        self.xmax       = self.dict["xmax"]
        self.xmin       = self.dict["xmin"]
        self.x, self.h  = np.linspace(self.xmin,
                                     self.xmax,
                                     self.Nx,
                                     retstep=True)

        self.Nt         = self.dict["Nt"]
        self.tmax       = self.dict["tmax"]
        self.t,self.dt  = np.linspace(0.0,
                                     self.tmax,
                                     self.Nt,
                                     retstep=True)


        self.a          = self.dict["a"]
        self.x0         = self.dict["x0"]
        self.gamma      = self.dict["gamma"]
        self.xb         = self.dict["xb"]

        self.softening  = self.dict["softening"]
        self.R          = self.dict["R"]
        self.ABool      = self.dict["Abool"]
        self.Vsel       = self.dict["Vsel"]

        self.amp        = self.dict["amp"]
        self.w          = self.dict["w"]

        self.Mask()
        #self.HamSetUp()
        self.GroundState()
        self.psi=self.WaveFunction()

    def WaveFunction(self):
        psi = np.exp(-(self.x-self.x0)**2/(2*self.a**2))
        C = Math.Norm(psi,self.x)
        return psi.astype(complex)/C


    def Mask(self):
        self.mask = np.ones(len(self.x),dtype='float')
        for i in range(len(self.x)):
            if self.x[i]<(self.x[0]+self.xb):
                self.mask[i] = np.power(np.cos(np.pi*(self.x[i]- \
                               (self.x[0]+self.xb))*self.gamma/(2*self.xb)),1/8)
            if self.x[i]>(self.x[-1]-self.xb):
                self.mask[i] = np.power(np.cos(np.pi*(self.x[i]-(self.x[-1]- \
                               self.xb))*self.gamma/(2*self.xb)),1/8)
        return

    def HamSetUp(self):
        self.H = Ham(x = self.x,
                               N = self.Nx,
                               h = self.h,
                               t= self.t,
                               V = self.Vsel,
                               ABool = self.ABool,
                               R = self.R,
                               softening = self.softening,
                               tmax = self.tmax,
                               amp = self.amp,
                               w = self.w)
        return

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

    def GroundState(self):
        H = Ham(x = self.x,
              N = self.Nx,
              h = self.h,
              t= self.t,
              V = self.Vsel,
              ABool = False,
              R = self.R,
              softening = self.softening,
              tmax = self.tmax,
              amp = self.amp,
              w = self.w)

        psi = self.WaveFunction()
        Prop = Propagator(H,self.Nx,self.dt)
        psiG,GroundEnergy = GroundState(H,Prop,psi,self.x,1E-14)

        self.psiG = psiG
        self.GroundEnergy = GroundEnergy
        return

    def Evolution(self):

        Prop = CrankNicolson.Propagator(self.H,self.Nx,self.dt)
        a = np.real(H.H[0])
        b = np.real(H.H[1])
        c = H.H[2]
        w, v = linalg.eigh_tridiagonal(b,a,select='v',select_range=[-np.inf,0],check_finite=False)
