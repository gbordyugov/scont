from numpy import diag, ones, zeros, array, arange
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
        c[j,k] = (c4*c[j,k] - float(k)*c[j,k-1])/c3

      c[j,0] = c4*c[j,0]/c3

    c1 = c2

  return array(zip(*c))


def dmatrix(n, order, nsp):
  """
  returns a 1D finite-difference derivative matrix
  assumes no-flux boundaries
  Parameters:
  n      : number of points
  order  : order of the derivative
  nsp    : number of stencil points, must be odd
  """
  m = zeros((n, n), dtype=float)

  # first, fill the bulk of the matrix...
  x = arange(nsp, dtype=float) - (nsp-1.0)/2.0
  coefs = weights(0.0, x)[order]

  for (c, i) in zip(coefs, map(int, x)):
    m += c*diag(ones(n-abs(i)), i)

  # ... then the corners
  # imposing no-flux boundary conditions
  x = arange(nsp, dtype=float) - 1.0
  c0 = weights(0.0, x)[1]

  for j in xrange((nsp-1)/2):
    c = weights(float(j), x)[order]
    c = c - c[0]/c0[0]*c0

    r = m[j]            # take the row of the matrix
    r[0:nsp-1] = c[1:]

    r = m[-1-j, -1::-1] # reverse of the minus -j-th row of the matrix
    r[0:nsp-1] = c[1:]

  return m

def dm(n, length, perbc=False, nsp=9):
  m = dmatrix(n, 1, nsp)
  dx = float(length)/float(n-1)
  return lil_matrix(m/dx)


def ddm(n, length, perbc=False, nsp=9):
  m = dmatrix(n, 2, nsp)
  dx = float(length)/float(n-1)
  return lil_matrix(m/dx/dx)



#
# simple versions
# 
def dM(n, length, perbc=False, nsp=9):
  m = -diag(ones(n))
  m = m + diag(ones(n-1), 1)

  m[0,0] = 0.0
  m[0,1] = 0.0

  m[-1,-1] = 0.0
  m[-1,-2] = 0.0

  dx = float(length)/float(n-1)
  return lil_matrix(m/dx)


def dMplus(n, length, perbc=False, nsp=9):
  m =    -1.5*diag(ones(n))
  m = m + 2.0*diag(ones(n-1), 1)
  m = m - 0.5*diag(ones(n-2), 2)

  m[0,0] = 0.0
  m[0,1] = 0.0
  m[0,2] = 0.0

  m[-1,-1] = 0.0
  m[-2,-1] = 0.0
  m[-2,-2] = 1.5
  m[-2,-3] =-2.0
  m[-2,-4] = 0.5


  dx = float(length)/float(n-1)
  return lil_matrix(m/dx)

def ddM(n, length, perbc=False, nsp=9):
  m = -2.0*diag(ones(n))
  m = m + diag(ones(n-1),  1)
  m = m + diag(ones(n-1), -1)

  m[ 0, 1] = 2.0
  m[-1,-2] = 2.0

  dx = float(length)/float(n-1)
  return lil_matrix(m/dx/dx)


# DMatrix, DDMatrix = dm, ddm
DMatrix, DDMatrix = dMplus, ddM
