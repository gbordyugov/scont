#
# this is going to be a lightweight version of sector
# just the stuff the sector class would need
# no solving whatsoever
#
# note: no-flux boundary conditions are assumed in both r and theta

from scipy.sparse import kron as spkron, lil_matrix, coo_matrix

from numpy import zeros, eye, linspace, diag, arange, tile
from numpy.linalg import norm

from pickable import pickable
from dmatrix import DMatrix, DDMatrix
from dview import dview

class sector(pickable):
  parnames = ['nr', 'ntheta', 'nv', 'nsp', 'r', 'R', 'omega', 'arclength',
              'a', 'b', 'e', 'f', 'jac', 'D', 'dummy']
  def __init__(self, pars):
    self.__dict__.update(pars)

    self.data = zeros(self.shape3)


  #
  # geometry properties
  #
  @property
  def   ntot(self): return self.nr*self.ntheta*self.nv
  @property
  def shape1(self): return self.ntot
  @property
  def shape3(self): return (self.nr, self.ntheta, self.nv)
  @property
  def aspect(self): return float(self.ntheta)/float(self.nr)

  # radii
  @property
  def     r1(self): return self.r           # smaller radius
  @property
  def     r2(self): return self.r + self.R  # larger  radius

  # angle
  @property
  def  theta(self):  return  self.arclength/self.r2

  # deltas
  @property
  def     dr(self): return self.R    /float(self.nr     - 1)
  @property
  def dtheta(self): return self.theta/float(self.ntheta - 1)


  #
  # data properties
  # 
  @property
  def   flat(self): return self.data.reshape(self.shape1)
  @property
  def      u(self): return self.data[...,0]
  @property
  def      v(self): return self.data[...,1]

  @property
  def   pars(self): return dview(self.__dict__, self.parnames)

  @property
  def trans1(self): return self.drmatrix()    *self.flat
  @property
  def trans2(self): return self.dthetamatrix()*self.flat




  def drmatrix(self):
    dr = DMatrix(self.nr, self.R, False, self.nsp)
    return spkron(spkron(dr, eye(self.ntheta)), eye(self.nv))


  def dthetamatrix(self):
    dt = DMatrix(self.ntheta, self.theta, False, self.nsp)
    return spkron(spkron(eye(self.nr), dt), eye(self.nv))


  def lapmatrix(self):
    """ computes and returns the laplacian matrix """
    dr  =  DMatrix(self.nr,     self.R,     False, self.nsp)
    drr = DDMatrix(self.nr,     self.R,     False, self.nsp)
    dtt = DDMatrix(self.ntheta, self.theta, False, self.nsp)

    radii = linspace(self.r1, self.r2, self.nr, endpoint=True)

    # one over r factor in front of the first radial derivative
    dr = dr.todense()
    for (row, r) in zip(dr, radii):
      row /= r
    dr = lil_matrix(dr)

    r = radii

    l1 = spkron(drr, eye(self.ntheta))
    l2 = spkron(dr,  eye(self.ntheta))
    l3 = spkron(diag(1.0/r/r, 0), dtt)

    return spkron(l1+l2+l3, self.D)


  def rhs(self):
    dtheta = self.dthetamatrix() * self.flat
    lap    = self.lapmatrix()    * self.flat
    f = (self.f(self.data, self.pars)).reshape(self.shape1)

    return f + self.omega*dtheta + lap


  def jacobian(self):
    Dtheta = self.dthetamatrix()
    lap    = self.lapmatrix()
    
    mat = lap + self.omega*Dtheta

    j = self.jac(self.data, self.pars).transpose((2, 3, 0, 1))
    n = self.nv*self.nv*self.nr*self.ntheta
    j = j.reshape(n)
    j = coo_matrix((j, self.jacindices()))

    return (mat + j).tolil()


  def jacindices(self):
    """ computes indices of the jacobian of the kinetics """
    nv, nr, ntheta = self.nv, self.nr, self.ntheta

    coi = arange(nv).repeat(nv)
    coj = tile(arange(nv), nv)

    coI = (arange(nr*ntheta)*nv).repeat(nv*nv)
    coJ = (arange(nr*ntheta)*nv).repeat(nv*nv)

    for i in xrange(len(coi)): ## len(coi) == nv*nv !
      coI[i::nv*nv] += coi[i]
      coJ[i::nv*nv] += coj[i]

    return coI, coJ
