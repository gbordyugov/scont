from copy import deepcopy

from numpy import zeros, append as npappend, dot, ones, savetxt, array
from numpy.linalg import norm

from continuation import continuation
from finger import finger
from sector import sector
from fhn import FHNNonlinearity, FHNJacobian
from matrix import sparse_matrix, augmented_matrix
from tip import FindTip as tip
from f2s import f2s

if False and __name__ == '__main__':
  f = finger.load('fingers/start.finger.11')
  f.dummy = 0.0
  f.nsp = 5
  s = f2s(f, 1.0e6, {'nsp': 5, 'dummy': 0.0})


def ucont(obj, par1name, par2name, par3name, nsteps, ds, zfuncs=[]):
  solution = obj
  obj = deepcopy(obj) # to preserve the old object

  
  def object2vector(obj):
    """ a helper function to map obj --> (u, p)"""
    u = zeros(len(obj.flat)+2)
    u[:-2] = obj.flat[:]
    u[ -2] = obj.pars[par2name]
    u[ -1] = obj.pars[par3name]
    p      = obj.pars[par1name]
    return u, p
  
  def vector2object(obj, u, p):
    """ a helper function to map (u, p) --> obj"""
    obj.flat[:]        = u[:-2]
    obj.pars[par2name] = u[ -2]
    obj.pars[par3name] = u[ -1]
    obj.pars[par1name] = p
  
  def dfdpar(obj, u, p, dpar, d = 1.0e-3):
    """ a helper function fo numerically approximating the partial
        derivative of obj with respect to obj.pars[dpar], evaluated at
        (u, p)"""
    vector2object(obj, u, p)
  
    obj.pars[dpar] += d/2.0; rhs2 = obj.rhs()
    obj.pars[dpar] -= d    ; rhs1 = obj.rhs()
    obj.pars[dpar] += d/2.0
  
    return (rhs2 - rhs1)/d
  
  
  def f(u, p):
    vector2object(obj, u, p)
    return npappend(obj.rhs(), [0.0, 0.0])
  
  
  def dfdx(u, p):
    vector2object(obj, u, p)
    dfdpar2 = dfdpar(obj, u, p, par2name)
    dfdpar3 = dfdpar(obj, u, p, par3name)
    
    trans1 = obj.trans1
    trans2 = obj.trans2
  
    t = tip(obj.u, obj.v)[0]
    i, j = int(t[0]), int(t[1])
  
    mask = ones(obj.shape3, dtype=bool)
    nx, ny, nz = mask.shape
  
    k = 50
    imin, imax = max(i-k, 0), min(i+k, nx)
    jmin, jmax = max(j-k, 0), min(j+k, ny)
    
    mask[imin:imax, jmin:jmax, :] = False
  
    mask = mask.reshape(obj.shape1)
    trans1[mask] = 0.0
    trans2[mask] = 0.0
  
    j = sparse_matrix(obj.jacobian())
    j = augmented_matrix(j, dfdpar2, trans1, 0.0)
    j = augmented_matrix(j, dfdpar3, trans2, 0.0)
  
    return j
  
  
  def dfdp(u, p):
    return npappend(dfdpar(obj, u, p, par1name), [0.0, 0.0])
  
  
  
  #
  # main entry point
  # 

  fname = 'fort.7'
  fort7 = open(fname, 'w')
  fort7.write('#\n# %s   %s   %s   norm\n#\n'%(par1name, par2name, par3name))
  fort7.close()

  branch = []
  
  def callback(u, par1):
    par2 = u[-2]
    par3 = u[-1]
  
    nrm = norm(u[:-2])

    print '%s: %e, %s: %e, %s: %e, norm: %e'\
        %(par1name, par1, par2name, par2, par3name, par3, nrm)

    lst = [par1, par2, par3, nrm]
    branch.append(lst)
  
    fort7 = open(fname, 'a')
    savetxt(fort7, array(lst).reshape(1,4), fmt='%.12e')
    fort7.close()
  
    flat = u[:-2].reshape(obj.shape3)
    t = tip(flat[...,0], flat[...,1])[0]
    print 'tip coordinates:', t[0], t[1]

    vector2object(solution, u, par1)
    solution.save('objects/step.object')
  
    return 0 # continue continuation
  
  
  u, p = object2vector(obj)
  
  try:
    u, p = continuation(f, dfdx, dfdp, u, p, nsteps, ds, callback, zfuncs)
  except KeyboardInterrupt:
    print 'bla-bla'
    # raise

  solution.save('objects/fc.object')

  return branch, solution
