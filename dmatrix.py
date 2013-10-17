from numpy import diag, ones, zeros, array
from scipy.sparse import lil_matrix

def weights(z, x):# , n, m):
  """ this is the famous Fornberg's weights routine to compute the coeffs
      of finite differences approximation.

      Parameters:
      z    : the x-coordinate where the approximation is needed
      x    : the array of x-coordinates of the grid points """

  n = len(x)-1
  m = len(x)-1

  c1 = 1.0
  c4 = x[0]-z

  c = zeros((n+1, m+1))

  c[0,0] = 1.0

  for i in xrange(1, n+1):
    mn = min(i, m)
    c2 = 1.0
    c5 = c4
    c4 = x[i]-z

    for j in xrange(i):
      c3 = x[i]-x[j]
      c2 = c2*c3

      if j == i-1:
        for k in xrange(mn,0,-1):
          c[i,k] = c1*(k*c[i-1,k-1] - c5*c[i-1,k])/c2
        c[i,0] = -c1*c5*c[i-1,0]/c2

      for k in xrange(mn,0,-1):
        c[j,k] = (c4*c[j,k] - k*c[j,k-1])/c3

      c[j,0] = c4*c[j,0]/c3

    c1 = c2

  return array(zip(*c))


def DMatrixFourthOrder(n, l, perbc):
  ## 1/12, -2/3, 0, 2/3, -1/12
  C = [1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0]
  a = 2.0/3.0
  b = 1.0/12.0
  m = a*diag(ones(n-1), 1) - a*diag(ones(n-1), -1) \
    - b*diag(ones(n-2), 2) + b*diag(ones(n-2), -2)

  if perbc:
    m[0,-2], m[0,-1] = 1.0/12.0, -2.0/3.0
    m[1,-1] = 1.0/12.0
    m[-1,0], m[-1,2] = 2.0/3.0, -1.0/12.0
    m[-2,0] =-1.0/12.0
  else:
    m[ 0,:] = 0.0
    m[-1,:] = 0.0
    m[1,  1] += 1.0/12.0
    m[-2,-2] +=-1.0/12.0

    n = n - 1

  odx = 1.0/(float(l)/float(n))
  return lil_matrix(m*odx)


def DDMatrixFourthOrder(n, l, perbc):
  ## -1/12, 4/3, -5/2, 4/3, -1/12
  a = 1.0/12.0
  b = 4.0/3.0
  c = 5.0/2.0

  C = [-1.0/12.0, 4.0/3.0, -5.0/2.0, 4.0/3.0, -1.0/12.0]

  d, o = diag, ones
  m = -a*d(o(n-2), -2) + b*d(o(n-1), -1) + \
      (-c)*d(o(n), 0) + \
      b*d(o(n-1), 1) + (-a)*d(o(n-2), 2)

  if perbc:
    m[0,-2], m[0,-1] = -1.0/12.0,  4.0/3.0
    m[1,-1] = -1.0/12.0
    m[-1,0], m[-1,2] =  4.0/3.0, -1.0/12.0
    m[-2,0] = -1.0/12.0
  else:
    m[1,  1] += -1.0/12.0
    m[-2,-2] += -1.0/12.0

    m[ 0,:],  m[-1,:]  =  0.0, 0.0
    m[ 0, 0], m[ 0, 1] = -2.0, 2.0
    m[-1,-2], m[-1,-1] =  2.0,-2.0
    n = n - 1

  odx2 = 1.0/(float(l)/float(n))**2
  return lil_matrix(m*odx2)


def DMatrixSecondOrder(n, l, perbc=False):
  m = diag(ones(n-1), 1) - diag(ones(n-1), -1)

  if perbc:
    m[0,-1], m[-1,0] =-1.0, 1.0
  else:
    m[0,1], m[-1,-2] = 0.0, 0.0
    n = n - 1

  odx = 1.0/(2.0*float(l)/float(n))
  return lil_matrix(m*odx)


def DDMatrixSecondOrder(n, l, perbc=False):
  m = diag(ones(n-1), 1) - 2.0*diag(ones(n), 0) + diag(ones(n-1), -1)

  if perbc:
    m[0,-1], m[-1,0] = 1.0, 1.0
  else:
    m[0,1], m[-1,-2] = 2.0, 2.0
    n = n - 1

  odx2 = 1.0/(float(l)/float(n))**2
  return lil_matrix(m*odx2)


# DMatrix, DDMatrix = DMatrixSecondOrder, DDMatrixSecondOrder
DMatrix, DDMatrix = DMatrixFourthOrder, DDMatrixFourthOrder
