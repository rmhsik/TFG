import numpy as np
from scipy import integrate
from potential import V

class Numerov:
    def __init__(self,xmin,xmax,N,ep=0):
        #Physical constants in a.u.
        self.me = 1 #Electron mass
        self.hbar = 1 #Planck's constant
        
        #Mesh parameters
        self.N = N #Number of points
        self.xmax = xmax #Maximum point for the domain
        self.xmin = xmin #Minimum point for the domain
        self.a = 0 #Convergence point
        #self.x, self.xl, self.xr = self.Mesh(self.xmin,self.xmax,self.a,self.N)
        
        self.ep = ep
        
    def Mesh(self):
        self.xl = np.linspace(self.xmin,self.a,self.N)
        self.xr = np.linspace(self.a,self.xmax,self.N)
        self.x = np.concatenate((self.xl,self.xr))
        self.h2 = (abs(self.xmax-self.xmin)/(2*self.N))**2
            
    def K2(self):
        self.Mesh()
        self.k2 = 2*self.me/self.hbar**2*(self.ep-V(self.x))
    
    def EvenWaveFunction(self):
        self.K2()
        
        psir = np.zeros(self.N)
        psil = np.zeros(self.N)
        
        psil[0]=0
        psil[1] = 1E-7
        psir[-1] = 0
        psir[-2] = 1E-7 # positive for even parity eigenfunctions
        for i in range(2,self.N):
            psil[i] = (2*(1-(5.0/12)*self.h2*self.k2[i-1])*psil[i-1]-(1+(1.0/12)*self.h2*self.k2[i-2])*psil[i-2])/(1+(1.0/12)*self.h2*self.k2[i])
        for i in range(3,self.N+1): #The negative indexes indicates that we start from right to the left
            psir[-i] = (2*(1-(5.0/12)*self.h2*self.k2[-i+1])*psir[-i+1]-(1+(1.0/12)*self.h2*self.k2[-i+2])*psir[-i+2])/(1+(1.0/12)*self.h2*self.k2[-i])
        self.PsiR = psir
        self.PsiL = psil
        return [psil,psir]
    
    def OddWaveFunction(self):
        self.K2()
        
        psir = np.zeros(self.N)
        psil = np.zeros(self.N)
        
        psil[0]=0
        psil[1] = 1E-7
        psir[-1] = 0
        psir[-2] = -1E-7 # negative for odd parity eigenfunctions
        for i in range(2,self.N):
            psil[i] = (2*(1-(5.0/12)*self.h2*self.k2[i-1])*psil[i-1]-(1+(1.0/12)*self.h2*self.k2[i-2])*psil[i-2])/(1+(1.0/12)*self.h2*self.k2[i])
        for i in range(3,self.N+1): #The minus in indexes indicates that we start from right to the left
            psir[-i] = (2*(1-(5.0/12)*self.h2*self.k2[-i+1])*psir[-i+1]-(1+(1.0/12)*self.h2*self.k2[-i+2])*psir[-i+2])/(1+(1.0/12)*self.h2*self.k2[-i])
        self.PsiR = psir
        self.PsiL = psil
        return [psil,psir]
     
    def normalize(self,psi):
        psi = np.concatenate((psi[0],psi[1]))
        psi2 = psi*psi
        C = integrate.simps(psi2)
        return C
    
    def LDerivate(self,psi,x):
        return (psi[x]-psi[x-1])/(np.sqrt(self.h2))
    
    def RDerivate(self,psi,x):
        return (psi[x+1]-psi[x])/(np.sqrt(self.h2))
    
    def WaveFunction(self):
        psi_even = self.EvenWaveFunction()
        psi_odd  = self.OddWaveFunction()
        
        psin_even = psi_even/np.sqrt(self.normalize(psi_even))
        psin_odd = psi_odd/np.sqrt(self.normalize(psi_odd))
        
        if (abs(psin_even[0][-1]-psin_even[1][0])<1E-3 and abs(self.LDerivate(psin_even[0],-1)-self.RDerivate(psin_even[1],0))<1E-2): #TODO: ADD FIRST DERIVATIVE CONDITION
            return psin_even
        elif (abs(psin_odd[0][-1]-psin_odd[1][0])<1E-3 and abs(self.LDerivate(psin_odd[0],-1)-self.RDerivate(psin_odd[1],0))<1E-2):
            return psin_odd
        else:
            #print("Non continuous wave function")
            return False
        #return psin_even,psin_odd
        
        
    def fEven(self,ep):
        self.ep = ep
        psi = self.EvenWaveFunction()
        psin = psi/np.sqrt(self.normalize(psi))
        return (psin[0][-1]-psin[1][0])+(self.LDerivate(psin[0],-1)-self.RDerivate(psin[1],0))

    def fOdd(self,ep):
        self.ep = ep
        psi = self.OddWaveFunction()
        psin = psi/np.sqrt(self.normalize(psi))
        return (psin[0][-1]-psin[1][0])+(self.LDerivate(psin[0],-1)-self.RDerivate(psin[1],0))

    def ShootingEven(self,eps1,eps2,deps):
        ep0 = eps1
        ep1 = eps2
        while abs(ep1-ep0)>deps:
            ep2 = ep1-(ep1-ep0)/(self.fEven(ep1)-self.fEven(ep0))*self.fEven(ep1)
            ep1 = ep2
            ep0 = ep1
        return ep1

    def ShootingOdd(self,eps1,eps2,deps):
        ep0 = eps1
        ep1 = eps2
        while abs(ep1-ep0)>deps:
            ep2 = ep1-(ep1-ep0)/(self.fOdd(ep1)-self.fOdd(ep0))*self.fOdd(ep1)
            ep1 = ep2
            ep0 = ep1
        return ep1

    def Shooting(self,eps,deps,delta,n):
        valid_ep = []
        eps1 = eps
        eps2 = eps1 + delta
        while len(valid_ep)<20:
            epOdd = self.ShootingOdd(eps1,eps2,deps)
            epEven = self.ShootingEven(eps1,eps2,deps)
            self.ep = epOdd
            psiOdd = self.WaveFunction()
            self.ep = epEven
            psiEven = self.WaveFunction()

            if type(psiOdd) != bool:
                valid_ep.append(epOdd)
                #print("odd")
                #print(epOdd)

            if type(psiEven) != bool :
                valid_ep.append(epEven)
                #print("even")
                #print(epEven)
            eps1 = eps2
            eps2 = eps1 + delta

        return valid_ep