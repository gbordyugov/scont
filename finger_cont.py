from numpy import zeros, append as npappend, dot, zeros_like, ones

from continuation import continuation
from finger import finger
from fhn import FHNNonlinearity, FHNJacobian
from matrix import sparse_matrix, augmented_matrix
from tip import FindTip as tip

f = finger.load('fingers/critical.finger')


def ue2finger(f, u, e):
  f.flat[:] = u[:-2]
  f.vx      = u[ -2]
  f.vy      = u[ -1]
  f.e       = e



def rhs(u, e):
  ue2finger(f, u, e)
  return npappend(f.rhs(), [0.0, 0.0])



def dfdpar(f, u, e, parname, d = 1.0e-6):
  ue2finger(f, u, e)

  dic = f.__dict__

  dic[parname] += d/2.0
  rhs2 = f.rhs()

  dic[parname] -= d
  rhs1 = f.rhs()

  dic[parname] += d/2.0

  return (rhs2 - rhs1)/d



def dfdx(u, e):
  ue2finger(f, u, e)
  dfdvx = dfdpar(f, u, e, 'vx')
  dfdvy = dfdpar(f, u, e, 'vy')
  
  dfdx = f.dxmatrix()*f.flat
  dfdy = f.dymatrix()*f.flat

  # constrain it close to the area around the tip
  if True:
    t = tip(f.u, f.v)[0]
    i, j = int(t[0]), int(t[1])

    dfdx     =     dfdx.reshape(f.shape3)
    dfdy = dfdy.reshape(f.shape3)

    k = 20
    mask = ones(f.shape3, dtype=bool)
    nx, ny, nz = mask.shape

    imin, imax = max(i-k, 0), min(i+k, nx)
    jmin, jmax = max(j-k, 0), min(j+k, ny)
    
    mask[imin:imax, jmin:jmax, :] = False

    dfdx[mask] = 0.0
    dfdy[mask] = 0.0
  
    dfdx = dfdx.reshape(f.shape1)
    dfdy = dfdy.reshape(f.shape1)

  j = sparse_matrix(f.jacobian())
  j = augmented_matrix(j, dfdvx, dfdx, 0.0)
  j = augmented_matrix(j, dfdvy, dfdy, 0.0)

  return j


def dfdp(u, e):
  ue2finger(f, u, e)
  return npappend(dfdpar(f, u, e, 'e'), [0.0, 0.0])


fname = 'finger.fort.7'
fort7 = open(fname, 'w')
fort7.close()

solution = []

def callback(u, e):
  vx = u[-2]
  vy = u[-1]
  print 'e =', e, ', vx =', vx, ', vy =', vy

  solution.append([e, vx, vy])

  file = open(fname, 'a')
  file.write(' '.join(map(str, [e, vx, vy]))+'\n')
  file.close()

  flat = u[:-2].reshape(f.shape3)
  t = tip(flat[...,0], flat[...,1])[0]
  print 'tip coordinates:', t[0], t[1]

  return 0 # continue continuation



u = zeros(len(f.flat)+2)
u[:-2] = f.flat[:]
u[ -2] = f.vx
u[ -1] = f.vy
e = f.e

def go():
  global u, e
  u, e = continuation(rhs, dfdx, dfdp, u, e, 1000, 2.5, callback) 
  ue2finger(f, u, e)
