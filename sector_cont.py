from numpy import zeros, append as npappend, dot

from scont import continuation
from sector import sector
from fhn import FHNNonlinearity, FHNJacobian
from matrix import sparse_matrix, augmented_matrix
from tip import FindTip as tip

s = sector.load('sectors/150.sector')


def ue2sector(s, u, e):
  s.flat[:] = u[:-2]
  s.r       = u[ -2]
  s.omega   = u[ -1]
  s.e       = e



def f(u, e):
  ue2sector(s, u, e)
  dsdr     = s.drmatrix()    *s.flat
  dsdtheta = s.dthetamatrix()*s.flat

  return npappend(s.rhs(), [0.0, 0.0])



def dfdpar(s, u, e, par, d = 1.0e-3):
  ue2sector(s, u, e)

  dic = s.__dict__

  dic[par] += d/2.0
  rhs2 = s.rhs()

  dic[par] -= d
  rhs1 = s.rhs()

  dic[par] += d/2.0

  return (rhs2 - rhs1)/d



def dfdx(u, e):
  ue2sector(s, u, e)

  dfdr = dfdpar(s, u, e, 'r')
  dfdo = dfdpar(s, u, e, 'omega')
  
  dsdr     = s.drmatrix()    *s.flat
  dsdtheta = s.dthetamatrix()*s.flat
  
  j = sparse_matrix(s.jacobian())
  j = augmented_matrix(j, dfdr, dsdr,     0.0)
  j = augmented_matrix(j, dfdo, dsdtheta, 0.0)

  return j


def dfdp(u, e):
  ue2sector(s, u, e)
  return npappend(dfdpar(s, u, e, 'e'), [0.0, 0.0])


u = zeros(len(s.flat)+2)
u[:-2] = s.flat[:]
u[ -2] = s.r
u[ -1] = s.omega
e = s.e


def callback(u, e):
  r = u[-2]
  o = u[-1]
  print 'e =', e, ', r =', r, ', omega =', o


def go():
  global u, e
  u, e = continuation(f, dfdx, dfdp, u, e, 10, 5.0, callback) 
  s.e = e
  s.flat[:] = u[:-2]
  s.r       = u[ -2]
  s.omega   = u[ -1]
