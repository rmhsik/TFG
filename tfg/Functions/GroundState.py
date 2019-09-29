import numpy as np
from scipy import integrate
from Functions import Math

def GroundState(H,Prop,psi0,x,delta):
    dt = -1j*0.01
    Prop.dt = dt
    psi0 = psi0
    eps = 1
    ep0 = Math.EigenEnergy(psi0,H,x)
    while (eps>delta):
        psi1 = Prop.Propagate(psi0)
        norm = Math.Norm(psi1,x)
        psi1 = psi1/np.sqrt(norm)
        ep1 = Math.EigenEnergy(psi1,H,x)
        eps = abs(ep1-ep0)
        ep0 = ep1
        psi0 = psi1
    return (psi1,ep1)
