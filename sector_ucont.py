from numpy import zeros, append as npappend, dot, ones
from numpy.linalg import norm

from continuation import continuation
from finger import finger
from sector import sector
from fhn import FHNNonlinearity, FHNJacobian
from matrix import sparse_matrix, augmented_matrix
from tip import FindTip as tip
from f2s import f2s

f = finger.load('fingers/start.finger.11')

s = f2s(f, 1.0e6, {'nsp': 5, 'dummy': 0.0})

def sector_ucont(s, par1name, par2name, par3name, nsteps, ds):
  def ue2sector(s, u, p):
    """ a helper function to map (u, p) --> s"""
    s.flat[:]        = u[:-2]
    s.pars[par2name] = u[ -2]
    s.pars[par3name] = u[ -1]
    s.pars[par1name] = p
  
  
  def dfdpar(s, u, p, dpar, d = 1.0e-3):
    """ a helper function fo numerically approximating the partial
        derivative of s with respect to s.pars[dpar], evaluated at (u, p)"""
    ue2sector(s, u, p)
  
    s.pars[dpar] += d/2.0; rhs2 = s.rhs()
    s.pars[dpar] -= d    ; rhs1 = s.rhs()
    s.pars[dpar] += d/2.0
  
    return (rhs2 - rhs1)/d
  
  
  def f(u, p):
    ue2sector(s, u, p)
    return npappend(s.rhs(), [0.0, 0.0])
  
  
  
  def dfdx(u, p):
    ue2sector(s, u, p)
    dfdpar2 = dfdpar(s, u, p, par2name)
    dfdpar3 = dfdpar(s, u, p, par3name)
    
    dsdr     = s.drmatrix()    *s.flat
    dsdtheta = s.dthetamatrix()*s.flat
  
    if True:
      t = tip(s.u, s.v)[0]
      i, j = int(t[0]), int(t[1])
  
      mask = ones(s.shape3, dtype=bool)
      nx, ny, nz = mask.shape
  
      k = 20
      imin, imax = max(i-k, 0), min(i+k, nx)
      jmin, jmax = max(j-k, 0), min(j+k, ny)
      
      mask[imin:imax, jmin:jmax, :] = False
  
      mask = mask.reshape(s.shape1)
      dsdr    [mask] = 0.0
      dsdtheta[mask] = 0.0
  
    j = sparse_matrix(s.jacobian())
    j = augmented_matrix(j, dfdpar2, dsdr,     0.0)
    j = augmented_matrix(j, dfdpar3, dsdtheta, 0.0)
  
    return j
  
  
  def dfdp(u, p):
    return zeros(s.ntot+2, dtype=float)
    # return npappend(dfdpar(s, u, p, par1name), [0.0, 0.0])
  
  
  fname = 'fort.7'
  fort7 = open(fname, 'w')
  fort7.close()
  
  solution = []
  
  def callback(u, p):
    par2 = u[-2]
    par3 = u[-1]
    print par1name + ': ' + str(p   ) + ', ',\
          par2name + ': ' + str(par2) + ', ',\
          par3name + ': ' + str(par3) + ', '
  
    solution.append([p, par2, par3])
  
    fort7 = open(fname, 'a')
    fort7.write(' '.join(map(str, [p, par2, par3]))+'\n')
    fort7.close()
  
    flat = u[:-2].reshape(s.shape3)
    t = tip(flat[...,0], flat[...,1])[0]
    print 'tip coordinates:', t[0], t[1]
  
    return 0 # continue continuation
  
  
  
  u = zeros(len(s.flat)+2)
  u[:-2] = s.flat[:]
  u[ -2] = s.pars[par2name]
  u[ -1] = s.pars[par3name]
  p      = s.pars[par1name]
  
  u, p = continuation(f, dfdx, dfdp, u, p, nsteps, ds, callback) 
  ue2sector(s, u, p)
  s.save('sectors/ic.sector')

  return s
