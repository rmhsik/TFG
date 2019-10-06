# distutils: language=c++
# cython: language_level=3

'''Computes the multiplication of a tridiagonal matrix,
given only its diagonals as vectorrs, with a column vector.
It is otimized with cython'''

import numpy as np

def TriDot(a,b,c,d):
    return cTriDot(a,b,c,d)

cdef double complex[:] cTriDot(double complex[:] aa,
                              double complex[:] bb,
                              double complex[:] cc,
                              double complex[:] d):
    '''a = diagonal, b = supradiaognal, c =subdiagonal
       d = column vector.
       len(a) = len(d) = n
       len(b) = len(c) = n-1'''
    cdef double complex[:] b = bb
    cdef double complex[:] c = np.append(0,cc)
    cdef double complex[:] a = np.append(aa,0)
    cdef int n = len(b)
    cdef int i
    
    cdef double complex[:] x = np.zeros(n,dtype=complex)    
    
    x[0] = b[0]*d[0] + a[0]*d[1]
    x[-1] = c[-1]*d[-2] + b[-1]*d[-1]
    
    for i in range(1,n-1):
        x[i] = c[i]*d[i-1] + b[i]*d[i]+a[i]*d[i+1]
    
    return x