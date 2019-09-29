import numpy as np
from scipy import integrate

def derivate(f,x):
    df = np.zeros(np.shape(f))

    if np.shape(f) == np.shape(x):
        for i in range(len(f)):
            try:
                df[i]=(f[i+1]-f[i])/(x[i+1]-x[i])
            except:
                df[i] = df[i-1]

    return df

def diag(N):
    #################################################
    #   Return a NxN matrix with ones in the diagonal
    #################################################

    if type(N) == int and N>0:

        M = np.zeros((N,N))
        for i in range(N):
            M[i,i] = 1

        return M
    else:
        print("N must be int and greater than 0 ")

def sdiag(N,d):
    ##################################################
    #   Return a NxN matrix with ones in the d-th
    #   diagonal
    ##################################################

    if type(N) == int and N > 0 and abs(d) < N:
        M = np.zeros((N,N))
        for i in range(N):
            try:
                if (i+d) >= 0:
                    M[i,i+d] = 1
                    #print(str(i)+','+str(i+d))
            except:
                pass

        return M
    else:
        print("N must be int, greater than 0 and d must be lesser than N")

def EigenEnergy(psi,H,x):
    ##################################################
    #   Return the eigenenegy given the hamiltonian
    #   and the eigenfunction, aka wavefunction
    ##################################################

    integ = np.conjugate(psi)*np.dot(H.H,psi)
    return integrate.simps(integ,x)

def Norm(psi,x):
    ##################################################
    #   Return the normalization factor rootsquared
    #   diagonal
    ##################################################

    return np.sqrt(integrate.simps(np.conjugate(psi)*psi,x))
