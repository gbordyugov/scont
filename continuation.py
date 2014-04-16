from numpy import array, zeros, hstack, dot, zeros_like, ndarray
from numpy.linalg import norm
from matrix import dense_matrix, sparse_matrix, augmented_matrix,\
                   base_matrix
from scipy.sparse import issparse

from sys import exit

tol = 1.0e-8
itmx = 9
nwtn = 4

dsmin = 1.0e-9
dsmax = 5.0e4

norm_explosion = 1.0e3

# by how much reduce/increase the step size
step_factor = 1.5



def continuation(f, dfdx, dfdp, x0, p0, nsteps, ds, callback=None,
                 zfuncs=None):
  """
  the main function for performing continuation

  f        : the nonlinearity
  dfdx     : the Jacobian matrix of the system, can be either:
              - a dense matrix, i.e. numpy.array object
              - a sparse matrix
              - a matrix.base_matrix or its subclasses object
  dfdp     : the derivative of f with respect to p
  x0       : initial solution
  p0       : initial value of parameter
  nstesp   : number of continuation steps to be taken
  ds       : initial step length
  callback : a function that will be called as callback(x, p)
             upon each successful continuation step
  zfuncs   : a list of functions, zero of which defines additional
             output points
  """

  # first perform some tests
  if len(x0) != len(f(x0, p0)):
    print 'mismatch of x and f(x, p) dimensions, exiting'
    exit(-1)

  if len(x0) != len(dfdp(x0, p0)):
    print 'mismatch of x and dfdp(x, p) dimensions, exiting'
    exit(-1)


  j = dfdx(x0, p0)
  if isinstance(j, list):
    j = array(j)

  shape = j.shape
  if shape[0] != shape[1]:
    print 'dfdx(x0, p) is not a square matrix, exiting'
    exit(-1)

  if shape[0] != len(x0):
    print 'dimension of dfdx(x, p) does not match dimension of x, exiting'
    exit(-1)

  ndim = len(x0)


  # a couple of helper functions
  def compute_z(x, p):
    """ returns values of zero functions """
    if zfuncs is not None:
      return [f(x, p) for f in zfuncs]


  def build_ext_rhs(x, p, x0, p0, xp, pp, ds):
    """ computes right-hand side of the extended system """
    return hstack([f(x, p), dot(x-x0, xp) + (p-p0)*pp - ds])


  def build_ext_matrix(dfdx, x, p, tv):
    """ builds the Jacobian matrix of the extended system """
    if   isinstance(dfdx, base_matrix):  m = dfdx
    elif isinstance(dfdx, list):         m = dense_matrix(dfdx)
    elif isinstance(dfdx, ndarray):      m = dense_matrix(dfdx)
    elif   issparse(dfdx):               m = sparse_matrix(dfdx)
    else:
      print 'unknown type of Jacobian matrix, exiting'
      exit(-1)

    return augmented_matrix(m, dfdp(x,p), tv[:-1], tv[-1])




  def compute_tangent_vector(dfdx, x, p, old=None):
    """ computes tangent vector along the solution branch """
    if old is not None:
      tv = old
    else:
      tv = zeros(ndim+1)
      tv[-1] = 1.0

    m = build_ext_matrix(dfdx, x, p, tv)

    b = zeros(ndim+1)
    b[-1] = 1.0
    tv = (m.factorize())(b)

    return tv/norm(tv)


  def try_continuation_step(x0, p0, jac0, xp, pp, ds):
    jac = jac0

    z0 = compute_z(x0, p0)

    x = x0 + xp*ds
    p = p0 + pp*ds

    nrm = norm(build_ext_rhs(x, p, x0, p0, xp, pp, ds))
    print '   initial norm:', nrm
    nstep = 0

    while nrm > tol and nrm < norm_explosion and nstep < itmx:
      if nstep > 0:                  # otherwise take jac from the 
        jac = dfdx(x, p)             # computation of the tangent vector

      # perform a solve of the Newton's method
      m  = build_ext_matrix(jac, x, p, tv)
      b  = build_ext_rhs   (x, p, x0, p0, xp, pp, ds)
      du = (m.factorize())(-b)

      x += du[:-1] # Newton's update of the solution
      p += du[ -1]
      nrm = norm(build_ext_rhs(x, p, x0, p0, xp, pp, ds))
      print 'nstep', nstep, ',  norm:', nrm

      nstep += 1

    z1 = compute_z(x, p)

    return x, p, jac, nrm, nstep

  #
  # main entry point here
  # 


  jac = dfdx(x0, p0)
  tv = compute_tangent_vector(jac, x0, p0)

  xp = tv[:-1]
  pp = tv[ -1]

  cstep = 0
  while cstep < nsteps:
    print 'cstep:', cstep, ', ds:', ds
    cstep += 1

    z0 = compute_z(x0, p0)

    x, p, jac, nrm, nstep = try_continuation_step(x0, p0, jac, xp, pp, ds)

    if nrm <= tol: # converged
      if callback is not None:
        if callback(x, p) != 0:
          return x, p

      z1 = compute_z(x, p)

      x0, p0 = x, p
      tv = compute_tangent_vector(jac, x, p, tv) # jac is already available
                                                 # and factorized
      xp = tv[:-1]
      pp = tv[ -1]

      if nstep <= itmx/3 and abs(ds*step_factor) < abs(dsmax):
        newds = ds*step_factor
        print 'increasing step to', newds
        ds = newds

      
    else: # not yet converged
      if abs(ds/step_factor) >= abs(dsmin):
        newds = ds/step_factor/5.0
        print 'reducing step to', newds
        ds = newds
        cstep = cstep-1
      else:
        print 'no convergence using minimum step size, returning'
        return x, p

  return x, p
