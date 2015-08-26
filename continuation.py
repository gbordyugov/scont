from numpy import array, zeros, hstack, dot, zeros_like, ndarray
from numpy.linalg import norm
from matrix import dense_matrix, sparse_matrix, augmented_matrix,\
                   base_matrix
from scipy.sparse import issparse

from sys import exit

tol = 1.0e-5
itmx = 5
nwtn = 4

dsmin = 1.0e-9
dsmax = 5.0e4

norm_explosion = 1.0e3

# by how much reduce/increase the step size
step_factor = 1.5



def continuation(f, dfdx, dfdp, x0, p0, nsteps, ds, callback=None,
                 zfuncs=[]):
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
    """ returns the product of the values of zero functions """
    return reduce(lambda x, y: x*y, [f(x, p) for f in zfuncs], 1.0)


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




  def compute_tangent_vector(x, p, oldtv=None):
    """ computes tangent vector along the solution branch """
    if oldtv is not None:
      tv = oldtv
    else:
      tv = zeros(ndim+1)
      tv[-1] = 1.0

    jac = dfdx(x, p)
    m = build_ext_matrix(jac, x, p, tv)

    b = zeros(ndim+1)
    b[-1] = 1.0
    tv = (m.factorize())(b)

    return tv/norm(tv), jac


  def costep(x0, p0, ds, oldtv=None):
    tv, jac = compute_tangent_vector(x0, p0, oldtv)
    xp = tv[:-1]
    pp = tv[ -1]

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

    return x, p, tv, nrm, nstep


  def secant(x0, p0, x1, p1, s):
    """ secant method to find zeros of z """
    print 'sign of z changed'

    def z(s):
      x, p, tv, nrm, nstep = costep(x0, p0, s)
      return compute_z(x, p), x, p

    s0 = 0.0
    s1 = s
    z0 = compute_z(x0, p0)
    z1 = compute_z(x1, p1)

    it = 0
    while it < itmx and abs((s1-s0)/s) > 1.0e-6:
      dzds = (z1 - z0)/(s1 - s0)
      news = s1 - z1/dzds
      z0 = z1
      s0 = s1
      s1 = news
      z1, x, p = z(s1)
      print 'z1:', z1
      it = it + 1

    print 'returning'
    return x, p



  
  
  #
  # main entry point here
  # 


  tv = None
  cstep = 0
  while cstep < nsteps:
    print 'continuation step:', cstep, ', ds:', ds
    cstep += 1

    z0 = compute_z(x0, p0)
    print 'z0:', z0

    x, p, tv, nrm, nstep = costep(x0, p0, ds, tv)

    if nrm <= tol: # converged
      z1 = compute_z(x, p)
      print 'z1:', z1

      if callback is not None:
        if callback(x, p) != 0:
          return x, p

      #
      # check if the sign of test functions has changed
      #
      if (z0 > 0.0 and z1 < 0.0) or (z0 < 0.0 and z1 > 0.0):
        x, p = secant(x0, p0, x, p, ds)
        return x, p

      if nstep <= itmx/2 and abs(ds*step_factor) < abs(dsmax):
        newds = ds*step_factor
        print 'increasing step to', newds
        ds = newds

      x0, p0 = x, p

      
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
