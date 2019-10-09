# distutils: language=c++
# cython: language_level=3
import numpy as np

def TDMAsolver(double complex[:] c,
               double complex[:] b,
               double complex[:] a,
               double complex[:] d):

    return cTDMAsolver(a,b,c,d)

cdef double complex[:] cTDMAsolver(double complex[:] a,
                                   double complex[:] b,
                                   double complex[:] c,
                                   double complex[:] d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    cdef int nf = len(d) # number of equations
    cdef int it,il
    cdef double complex mc
    cdef double complex[:] ac,bc,cc,dc,xc

    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    #ac = np.array(a)
    #bc = np.array(b)
    #cc = np.array(c)
    #dc = np.array(d)

    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1]
        dc[it] = dc[it] - mc*dc[it-1]

    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc
