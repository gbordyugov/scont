from numpy import array, zeros, sqrt, hstack, vstack, newaxis, dot
from numpy.random import rand
from numpy.linalg import solve, norm
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
  return [[2.0*x, 2.0*y],
          [1.0,   0.0]]  



def dfdp(u, p):
  return array([0.0, -1.0])



def continuation(f, dfdx, dfdp, x0, p0, nsteps, ds):
  """
  the main function for performing continuation
  """

  def compute_ext_rhs(x, p, x0, p0, xp, pp, ds):
    """ computes right-hand side of the extenden system """
    return hstack([f(x, p), dot(x-x0, xp) + (p-p0)*pp - ds])


  def build_ext_matrix(x, p, tv):
    """ builds the Jacobian matrix of the extenden system """
    m = hstack([dfdx(x, p), dfdp(x, p)[:,newaxis]])
    return vstack([m, tv])


  def compute_tangent_vector(x, p, old=None):
    """ computes tangent vector along the solution branch """
    if old is not None:
      tv = old
    else:
      tv = zeros(ndim+1)
      tv[-1] = 1.0

    m = build_ext_matrix(x, p, tv)

    b = zeros(ndim+1)
    b[-1] = 1.0
    tv = solve(m, b)
    return tv/norm(tv)


  res         = hstack([x0, p0])
  predictions = hstack([x0, p0])

  tv = compute_tangent_vector(x0, p0)
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
      m = build_ext_matrix(x, p, tv)

      b = compute_ext_rhs(x, p, x0, p0, xp, pp, ds)
      du = solve(m, -b)

      x += du[:-1] # Newton's correction
      p += du[ -1]
      nrm = norm(compute_ext_rhs(x, p, x0, p0, xp, pp, ds))
      print 'nstep', nstep, ',  norm:', nrm

      nstep += 1

    if nrm <= tol:
      print 'converged, stepsize:', ds

      x0, p0 = x, p
      res = vstack([res, hstack([x, p])])

      tv = compute_tangent_vector(x, p, tv)
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
