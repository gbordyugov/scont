from numpy import array, zeros, hstack, dot, zeros_like, ndarray
from numpy.linalg import norm
from matrix import dense_matrix, sparse_matrix, augmented_matrix,\
                   base_matrix
from scipy.sparse import issparse

tol = 1.0e-8
ntst = 9

dsmin = 1.0e-9
dsmax = 5.0e-1

# by how much reduce/increase the step size
step_factor = 1.5



def continuation(f, dfdx, dfdp, x0, p0, nsteps, ds, callback=None):
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
  def compute_ext_rhs(x, p, x0, p0, xp, pp, ds):
    """ computes right-hand side of the extended system """
    return hstack([f(x, p), dot(x-x0, xp) + (p-p0)*pp - ds])


  def build_ext_matrix(dfdx, x, p, tv):
    """ builds the Jacobian matrix of the extended system """
    if isinstance(dfdx, base_matrix):
      return augmented_matrix(dfdx,                dfdp(x,p), tv[:-1], tv[-1])
    elif isinstance(dfdx, list):
      return augmented_matrix(dense_matrix(dfdx),  dfdp(x,p), tv[:-1], tv[-1])
    elif isinstance(dfdx, ndarray):
      return augmented_matrix(dense_matrix(dfdx),  dfdp(x,p), tv[:-1], tv[-1])
    elif issparse(dfdx):
      return augmented_matrix(sparse_matrix(dfdx), dfdp(x,p), tv[:-1], tv[-1])
    else:
      print 'unknown type of Jacobian matrix, exiting'
      exit(-1)


  def compute_tangent_vector(dfdx, x, p, old=None):
    """ computes tangent vector along the solution branch """
    if old is not None:
      tv = old
    else:
      tv = zeros(ndim+1)
      tv[-1] = 1.0

    m = build_ext_matrix(dfdx, x, p, tv)
    # print m.shape

    b = zeros(ndim+1)
    b[-1] = 1.0
    tv = (m.factorize())(b)

    return tv/norm(tv)


  #
  # main entry point here
  # 

  jac = dfdx(x0, p0)
  tv = compute_tangent_vector(jac, x0, p0)

  xp = tv[:-1]
  pp = tv[ -1]

  cstep = 0
  while cstep < nsteps:
    print 'cstep:', cstep
    cstep += 1

    x = x0 + xp*ds # prediction
    p = p0 + pp*ds

    nrm = norm(compute_ext_rhs(x, p, x0, p0, xp, pp, ds))
    print '   initial norm:', nrm
    nstep = 0

    while nrm > tol and nstep < ntst:
      if nstep > 0:        # otherwise take jac from the computation of
        jac = dfdx(x, p)   # the tangent vector above

      # perform a solve of the Newton's method
      m = build_ext_matrix(jac, x, p, tv)
      b = compute_ext_rhs(x, p, x0, p0, xp, pp, ds)
      du = (m.factorize())(-b)

      x += du[:-1] # Newton's update of the solution
      p += du[ -1]
      nrm = norm(compute_ext_rhs(x, p, x0, p0, xp, pp, ds))
      print 'nstep', nstep, ',  norm:', nrm

      nstep += 1

    if nrm <= tol: # converged
      if callback is not None:
        callback(x, p)

      x0, p0 = x, p
      tv = compute_tangent_vector(jac, x, p, tv) # jac is already available
                                                 # and factorized
      xp = tv[:-1]
      pp = tv[ -1]

      if nstep <= ntst/2 and abs(ds*step_factor) < abs(dsmax):
        print 'increasing step to', ds*step_factor
        ds = ds*step_factor
      
    else: # not yet converged
      if abs(ds/step_factor) >= abs(dsmin):
        print 'reducing step to', ds/step_factor
        ds = ds/step_factor
        cstep = cstep-1
      else:
        print 'no convergence using minimum step size'
        return x, p


  return x, p
