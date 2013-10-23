from numpy import array, sqrt
from continuation import continuation


def f(u, p):
  x, y = u[0], u[1]
  return [x**2 + y**2 - 1.0, x - p]



def dfdx(u, p):
  x, y = u[0], u[1]
  return ([[2.0*x, 2.0*y],
           [1.0,   0.0]])



def dfdp(u, p):
  return [0.0, -1.0]


solution = [] # here, we are going to collect our solution

def callback(u, p):
  x, y = u[0], u[1]
  print 'x =', x, ' y =', y, ' p =', p
  solution.append([x, y, p])

  return 0 # continue continuation


def go():
  value = sqrt(2.0)/2.0
  p0 = value
  x0 = [value, value]
  nsteps = 15
  ds = -0.1
  return continuation(f, dfdx, dfdp, x0, p0, nsteps, ds, callback)

go()
solution = array(solution)
print "issue plot(solution[:,0], solution[:,1], 'o') to plot the solution"
