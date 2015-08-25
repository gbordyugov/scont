#
# this is going to be a lightweight version of sector
# just the stuff the sector class would need
# no solving whatsoever
#
# note: no-flux boundary conditions are assumed in both r and theta

from scipy.interpolate import interp2d

from scipy.sparse import kron as spkron, lil_matrix, coo_matrix

from numpy import zeros, eye, linspace, diag, arange, tile
from numpy.linalg import norm

from pickable import pickable
from dmatrix import DMatrix, DDMatrix
from dview import dview

class finger(pickable):
  parnames = ['nx', 'ny', 'nv', 'nsp', 'vx', 'vy', 'lx', 'ly',
              'a', 'b', 'e', 'f', 'jac', 'D', 'dummy']
  def __init__(self, pars):
    self.__dict__.update(pars)

    self.data = zeros(self.shape3)


  #
  # geometry properties
  #
  @property
  def   ntot(self): return self.nx*self.ny*self.nv
  @property
  def shape1(self): return self.ntot
  @property
  def shape3(self): return (self.nx, self.ny, self.nv)
  @property
  def aspect(self): return float(self.ny)/float(self.nx)

  # deltas
  @property
  def     dx(self): return self.lx/float(self.nx - 1)
  @property
  def     dy(self): return self.ly/float(self.ny - 1)


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
  def trans1(self): return self.dxmatrix()*self.flat
  @property
  def trans2(self): return self.dymatrix()*self.flat


  def dxmatrix(self):
    dx = DMatrix(self.nx, self.lx, False, self.nsp)
    return spkron(spkron(dx, eye(self.ny)), eye(self.nv))


  def dymatrix(self):
    dy = DMatrix(self.ny, self.ly, False, self.nsp)
    return spkron(spkron(eye(self.nx), dy), eye(self.nv))


  def lapmatrix(self):
    """ computes and returns the laplacian matrix """
    dxx = DDMatrix(self.nx, self.lx, False, self.nsp)
    dyy = DDMatrix(self.ny, self.ly, False, self.nsp)

    lapx = spkron(dxx, eye(self.ny))
    lapy = spkron(eye(self.nx), dyy)

    return spkron(lapx+lapy, self.D)


  def rhs_test(self):
    """ returns a flat vector """
    dx  = self.dxmatrix()  * self.flat
    dy  = self.dymatrix()  * self.flat
    lap = self.lapmatrix() * self.flat
    f = (self.f(self.data, self.pars)).reshape(self.shape1)

    return (f + self.vx*dx + self.vy*dy + lap, f, dx, dy, lap)

  def rhs(self):
    """ returns a flat vector """
    dx  = self.dxmatrix()  * self.flat
    dy  = self.dymatrix()  * self.flat
    lap = self.lapmatrix() * self.flat
    f = (self.f(self.data, self.pars)).reshape(self.shape1)

    return f + self.vx*dx + self.vy*dy + lap

  def rhs3(self):
    """ returns a shape3 vector """
    return self.rhs().reshape(self.shape3)


  def jacobian(self):
    dx  = self.dxmatrix()
    dy  = self.dymatrix()
    lap = self.lapmatrix()
    
    mat = lap + self.vx*dx + self.vy*dy

    j = self.jac(self.data, self.pars).transpose((2, 3, 0, 1))
    n = self.nv*self.nv*self.nx*self.ny
    j = j.reshape(n)
    j = coo_matrix((j, self.jacindices()))

    return (mat + j).tolil()


  def jacindices(self):
    """ computes indices of the jacobian of the kinetics """
    nv, nx, ny = self.nv, self.nx, self.ny

    coi = arange(nv).repeat(nv)
    coj = tile(arange(nv), nv)

    coI = (arange(nx*ny)*nv).repeat(nv*nv)
    coJ = (arange(nx*ny)*nv).repeat(nv*nv)

    for i in xrange(len(coi)): ## len(coi) == nv*nv !
      coI[i::nv*nv] += coi[i]
      coJ[i::nv*nv] += coj[i]

    return coI, coJ


  def reshape(self, nx, ny):
    new_data = zeros((nx, ny, self.nv))

    x = arange(self.nx, dtype=float)/self.nx
    y = arange(self.ny, dtype=float)/self.ny

    for v in xrange(self.nv):
      z = self.data[...,v].T
      f = interp2d(x, y, z, kind='quintic')
      for i in xrange(nx):
        for j in xrange(ny):
          new_data[i, j, v] = f(float(i)/nx, float(j)/ny)

    self.data = new_data
    self.nx = nx
    self.ny = ny
