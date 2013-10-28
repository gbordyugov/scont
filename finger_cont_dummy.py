from numpy import zeros, append as npappend

from continuation import continuation
from finger import finger
from fhn import FHNNonlinearity, FHNJacobian
from matrix import sparse_matrix, augmented_matrix
from tip import FindTip as tip

ic_fname = 'fingers/start.finger.11'
fc_fname = 'fingers/start.finger.dummy'

f = finger.load(ic_fname)
f.e = 0.3
f.nsp = 11


def ue2finger(f, u, dummy):
  f.flat[:] = u[:-2]
  f.b       = u[ -2]
  f.vy      = u[ -1]
  # dummy



def rhs(u, dummy):
  ue2finger(f, u, dummy)
  return npappend(f.rhs(), [0.0, 0.0])


def dfdpar(f, u, dummy, parname, d = 1.0e-6):
  ue2finger(f, u, dummy)

  dic = f.__dict__

  dic[parname] += d/2.0
  rhs2 = f.rhs()

  dic[parname] -= d
  rhs1 = f.rhs()

  dic[parname] += d/2.0

  return (rhs2 - rhs1)/d



def dfdx(u, dummy):
  ue2finger(f, u, dummy)
  dfdb  = dfdpar(f, u, dummy, 'b')
  dfdvy = dfdpar(f, u, dummy, 'vy')
  
  dfdx = f.dxmatrix()*f.flat
  dfdy = f.dymatrix()*f.flat

  j = sparse_matrix(f.jacobian())
  j = augmented_matrix(j, dfdb,  dfdx, 0.0)
  j = augmented_matrix(j, dfdvy, dfdy, 0.0)

  return j


def dfdp(u, dummy):
  ue2finger(f, u, dummy)

  return zeros(f.ntot+2, dtype=float)
  # return npappend(dfdpar(f, u, e, 'e'), [0.0, 0.0])


fname = 'finger.fort.7'
fort7 = open(fname, 'w')
fort7.close()

solution = []

def callback(u, dummy):
  b  = u[-2]
  vy = u[-1]
  print 'dummy =', dummy, ', b =', b, ', vy =', vy

  solution.append([dummy, b, vy])

  file = open(fname, 'a')
  file.write(' '.join(map(str, [dummy, b, vy]))+'\n')
  file.close()

  flat = u[:-2].reshape(f.shape3)
  t = tip(flat[...,0], flat[...,1])[0]
  print 'tip coordinates:', t[0], t[1]

  return 0 # continue continuation



u = zeros(len(f.flat)+2)
u[:-2] = f.flat[:]
u[ -2] = f.b
u[ -1] = f.vy
# e = f.e

dummy = 0.0
def go():
  global u, dummy
  u, dummy = continuation(rhs, dfdx, dfdp, u, dummy, 3, -2.5, callback) 
  ue2finger(f, u, dummy)
  f.save(fc_fname)
