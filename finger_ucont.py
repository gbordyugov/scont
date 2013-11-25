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
f.dummy = 0.0
f.nsp = 5

s = f2s(f, 1.0e6, {'nsp': 5, 'dummy': 0.0})

def finger_ucont(f, par1name, par2name, par3name, nsteps, ds):
  def ue2finger(f, u, p):
    """ a helper function to map (u, p) --> f"""
    f.flat[:]        = u[:-2]
    f.pars[par2name] = u[ -2]
    f.pars[par3name] = u[ -1]
    f.pars[par1name] = p
  
  
  def dfdpar(f, u, p, dpar, d = 1.0e-3):
    """ a helper function fo numerically approximating the partial
        derivative of f with respect to f.pars[dpar], evaluated at (u, p)"""
    ue2finger(f, u, p)
  
    f.pars[dpar] += d/2.0; rhs2 = f.rhs()
    f.pars[dpar] -= d    ; rhs1 = f.rhs()
    f.pars[dpar] += d/2.0
  
    return (rhs2 - rhs1)/d
  
  
  def F(u, p):
    ue2finger(f, u, p)
    return npappend(f.rhs(), [0.0, 0.0])
  
  
  
  def dfdx(u, p):
    ue2finger(f, u, p)
    dfdpar2 = dfdpar(f, u, p, par2name)
    dfdpar3 = dfdpar(f, u, p, par3name)
    
    dfdx = f.dxmatrix()*f.flat
    dfdy = f.dymatrix()*f.flat
  
    if True:
      t = tip(f.u, f.v)[0]
      i, j = int(t[0]), int(t[1])
  
      mask = ones(f.shape3, dtype=bool)
      nx, ny, nz = mask.shape
  
      k = 20
      imin, imax = max(i-k, 0), min(i+k, nx)
      jmin, jmax = max(j-k, 0), min(j+k, ny)
      
      mask[imin:imax, jmin:jmax, :] = False
  
      mask = mask.reshape(f.shape1)
      dfdx[mask] = 0.0
      dfdy[mask] = 0.0
  
    j = sparse_matrix(f.jacobian())
    j = augmented_matrix(j, dfdpar2, dfdx, 0.0)
    j = augmented_matrix(j, dfdpar3, dfdy, 0.0)
  
    return j
  
  
  def dfdp(u, p):
    return npappend(dfdpar(f, u, p, par1name), [0.0, 0.0])
  
  
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
  
    flat = u[:-2].reshape(f.shape3)
    t = tip(flat[...,0], flat[...,1])[0]
    print 'tip coordinates:', t[0], t[1]
  
    return 0 # continue continuation
  
  
  
  u = zeros(len(f.flat)+2)
  u[:-2] = f.flat[:]
  u[ -2] = f.pars[par2name]
  u[ -1] = f.pars[par3name]
  p      = f.pars[par1name]
  
  u, p = continuation(F, dfdx, dfdp, u, p, nsteps, ds, callback) 
  ue2finger(f, u, p)
  f.save('fingers/ic.finger')

  return f
