from numpy import array, sqrt
from matrix import dense_matrix, sparse_matrix, augmented_matrix

from scont import continuation


def f(u, p):
  x, y = u[0], u[1]
  return [x**2 + y**2 - 1.0, x - p]



def dfdx(u, p):
  x, y = u[0], u[1]
  return sparse_matrix([[2.0*x, 2.0*y],
                       [1.0,   0.0]])



def dfdp(u, p):
  return [0.0, -1.0]


def go():
  value = sqrt(2.0)/2.0
  p0 = value
  x0 = [value, value]
  nsteps = 15
  ds = -0.1
  return continuation(f, dfdx, dfdp, x0, p0, nsteps, ds) 
