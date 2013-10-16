from numpy import diag, ones
from scipy.sparse import lil_matrix


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
