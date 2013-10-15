from numpy import array, zeros, sqrt, hstack, vstack, newaxis, dot,\
                  zeros_like
from numpy.random import rand
from numpy.linalg import solve, norm
from matrix import dense_matrix, sparse_matrix, augmented_matrix
# from scipy.linalg import lu_factor, lu_solve

ndim = 2
tol = 1.0e-8
ntst = 9

dsmin = 1.0e-9
dsmax = 5.0e-1

# by how much reduce/increase the step size
step_factor = 1.5



def f(u, p):
  x, y = u[0], u[1]
  return [x**2 + y**2 - 1.0, x - p]



def dfdx(u, p):
  x, y = u[0], u[1]
  return dense_matrix([[2.0*x, 2.0*y],
                       [1.0,   0.0]])



def dfdp(u, p):
  return [0.0, -1.0]


def continuation(f, dfdx, dfdp, x0, p0, nsteps, ds):
  """
  the main function for performing continuation

  res, predictions = continuation(f, dfdx, dfdp, x0, p0, nsteps, ds),
  where:
  f    - the nonlinearity
  dfdx - the Jacobian matrix of the system
         expected to have a ``factorize'' method, returning a function
         that represents the inverse operator
  """

  # a couple of helper functions
  def compute_ext_rhs(x, p, x0, p0, xp, pp, ds):
    """ computes right-hand side of the extended system """
    return hstack([f(x, p), dot(x-x0, xp) + (p-p0)*pp - ds])


  def build_ext_matrix(dfdx, x, p, tv):
    """ builds the Jacobian matrix of the extended system """
    return augmented_matrix(dfdx, dfdp(x,p), tv[:-1], tv[-1])


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


  #
  # main entry point here
  # 
  res         = hstack([x0, p0])
  predictions = hstack([x0, p0])

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
    predictions = vstack([predictions, hstack([x, p])])

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
      x0, p0 = x, p
      res = vstack([res, hstack([x, p])])

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
        return res, predictions


  return res, predictions



def go():
  value = sqrt(2.0)/2.0
  p0 = value
  x0 = array([value, value])
  nsteps = 15
  ds = -0.1
  return continuation(f, dfdx, dfdp, x0, p0, nsteps, ds) 
