from numpy import zeros, append as npappend

from scont import continuation
from sector import sector
from fhn import FHNNonlinearity, FHNJacobian
from matrix import sparse_matrix, augmented_matrix

s = sector.load('sectors/150.sector')


def ue2sector(s, u, e):
  s.flat[:] = u[:-2]
  s.r       = u[ -2]
  s.omega   = u[ -1]
  s.e       = e



def f(u, e):
  ue2sector(s, u, e)
  return npappend(s.rhs(), [0.0, 0.0])



def dfdpar(s, u, e, par, d = 1.0e-3):
  ue2sector(s, u, e)

  s.__dict__[par] += d/2.0
  rhs2 = s.rhs()
  s.__dict__[par] -= d
  rhs1 = s.rhs()
  s.__dict__[par] += d/2.0

  return (rhs2 - rhs1)/d



def dfdx(u, e):
  ue2sector(s, u, e)

  dfdr = dfdpar(s, u, e, 'r')
  dfdo = dfdpar(s, u, e, 'omega')
  
  dsdr     = s.drmatrix()    *s.flat
  dsdtheta = s.dthetamatrix()*s.flat
  
  # print dfdr.shape, dfdo.shape, dsdr.shape, dsdtheta.shape

  j = sparse_matrix(s.jacobian())
  j = augmented_matrix(j,          dfdr,                dsdr,           0.0)
  j = augmented_matrix(j, npappend(dfdo, 0.0), npappend(dsdtheta, 0.0), 0.0)

  return j


def dfdp(u, e):
  ue2sector(s, u, e)
  return npappend(dfdpar(s, u, e, 'e'), [0.0, 0.0])


u = zeros(len(s.flat)+2)
u[:-2] = s.flat[:]
u[ -2] = s.r
u[ -1] = s.omega
e = s.e

ds = 0.1

def callback(u, e):
  r = u[-2]
  o = u[-1]
  print 'e =', e, ', r =', r, ', omega =', o

def go():
  global u, e
  u, e = continuation(len(u), f, dfdx, dfdp, u, e, 10, ds, callback) 
  s.e = e
  s.flat[:] = u[:-2]
  s.r       = u[ -2]
  s.omega   = u[ -1]
