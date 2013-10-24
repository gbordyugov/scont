from numpy import zeros, append as npappend, dot, zeros_like

from continuation import continuation
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




def dfdpar(s, u, e, parname, d = 1.0e-6):
  ue2sector(s, u, e)

  dic = s.__dict__

  dic[parname] += d/2.0
  rhs2 = s.rhs()

  dic[parname] -= d
  rhs1 = s.rhs()

  dic[parname] += d/2.0

  return (rhs2 - rhs1)/d



def dfdx(u, e):
  ue2sector(s, u, e)
  dfdr = dfdpar(s, u, e, 'r')
  dfdo = dfdpar(s, u, e, 'omega')
  
  dsdr     = s.drmatrix()    *s.flat
  dsdtheta = s.dthetamatrix()*s.flat

  # constrain it close to the area around the tip
  if True:
    t = tip(s.u, s.v)[0]
    i, j = int(t[0]), int(t[1])

    dsdr     =     dsdr.reshape(s.shape3)
    dsdtheta = dsdtheta.reshape(s.shape3)

    k = 20
    mask = zeros_like(dsdr, dtype=bool)
    nx, ny, nz = mask.shape

    imin, imax = max(i-k, nx), min(i+k, nx)
    jmin, jmax = max(j-k, ny), min(j+k, ny)
    
    mask[imin:imax, jmin:jmax, :] = True
    mask = ~mask

    dsdr    [mask] = 0.0
    dsdtheta[mask] = 0.0
  
    dsdr     = dsdr.reshape(s.shape1)
    dsdtheta = dsdtheta.reshape(s.shape1)

  j = sparse_matrix(s.jacobian())
  j = augmented_matrix(j, dfdr, dsdr,     0.0)
  j = augmented_matrix(j, dfdo, dsdtheta, 0.0)

  return j


def dfdp(u, e):
  ue2sector(s, u, e)
  return npappend(dfdpar(s, u, e, 'e'), [0.0, 0.0])


fname = 'fort.7'
fort7 = open(fname, 'w')
fort7.close()

solution = []

def callback(u, e):
  r = u[-2]
  o = u[-1]
  print 'e =', e, ', r =', r, ', omega =', o

  solution.append([e, r, o])

  f = open(fname, 'a')
  f.write(' '.join(map(str, [e, r, o]))+'\n')
  f.close()

  flat = u[:-2].reshape(s.shape3)
  t = tip(flat[...,0], flat[...,1])[0]
  print 'tip coordinates:', t[0], t[1]

  return 0 # continue continuation



u = zeros(len(s.flat)+2)
u[:-2] = s.flat[:]
u[ -2] = s.r
u[ -1] = s.omega
e = s.e

def go():
  global u, e
  u, e = continuation(f, dfdx, dfdp, u, e, 1000, 5.0, callback) 
  ue2sector(s, u, e)
