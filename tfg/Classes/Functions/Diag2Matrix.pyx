# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False

import numpy as np

def Diag2Matrix(double complex[:] a,
                double complex[:] b,
                double complex[:] c):
    return np.asarray(cDiag2Matrix(a,b,c))

cdef double complex[:,:] cDiag2Matrix(double complex[:] a,
                                       double complex[:] b,
                                       double complex[:] c):
    '''
        Create a NxN tridiagonal Matrix given its diaognals
    '''
    cdef int N = len(b)
    cdef double complex[:,:] Matrix
    cdef double complex[:] aa,bb,cc
    cdef int i,j

    bb = np.asarray(b)
    aa = np.append(a,0)
    cc = np.append(0,c)

    Matrix=np.zeros((N,N)).astype(complex)

    for i in range(N):
        Matrix[i,i] = bb[i]

        if i<N-1:
            Matrix[i,i+1] = aa[i]

        if i>0:
            Matrix[i,i-1] = cc[i]
    return Matrix
