from numpy import array, zeros_like, ones_like

def FHNNonlinearity(x, p):
  a = p['a']
  b = p['b']
  e = p['e']
  u, v = x[...,0], x[...,1]
  f = zeros_like(x)
  f[...,0] = (u-u*u*u/3.0 - v)/e
  f[...,1] = (u - a*v + b)    *e
  return f


def FHNJacobian(x, p):
  a = p['a']
  b = p['b']
  e = p['e']*ones_like(x[...,0]) # make it a vector!

  u = x[...,0]
  v = x[...,1]

  return array([[(1.0 - u*u)/e, -1.0/e],
                [e,             -a*e]])
  

# derivative of the RHS with respect to e
def FHNdE(x, p):
  a = p['a']
  b = p['b']
  e = p['e']

  u = x[...,0]
  v = x[...,1]

  d = zeros_like(x)

  d[...,0] =-(u - u*u*u/3.0 - v)/e/e
  d[...,1] = (u - a*v + b)
  return d




fhnParameters = {'a': 0.5, 'b': 0.88, 'e': 0.3} # for contracting
fhnC          = array([0.0, 1.45])
fhnDiffusion  = [[1.0, 0.0],
    [0.0, 0.0]]
fhnDim        = (225,   150)
fhnSize       = (75.0,  50.0)

def fhnFe(x, p):
  a = p['a']
  b = p['b']
  e = p['e0'] + p['e1']
  u, v = x[...,0], x[...,1]
  f = zeros_like(x)
  f[...,0] = (u-u*u*u/3.0 - v)/e
  f[...,1] = (u - a*v + b)    *e
  return f


def fhnJe(x, p):
  a = p['a']
  b = p['b']
  e = (p['e0'] + p['e1'])*ones_like(x[...,0]) # make it a vector!

  u = x[...,0]
  v = x[...,1]

  return array([[(1.0 - u*u)/e, -1.0/e],
    [e,             -a*e]])



