# the ultimate purpose of this small library of classes is the class for
# augmented matrices, which are basically nothing but a some ``old'' matrix
# extended by a column, a row and a number in the lower right corner

from numpy import dot, zeros_like, zeros, array
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized
from scipy.linalg import lu_factor, lu_solve

#
# a base class for matrices
#

class base_matrix(object):
  def __init__(self, *args):
    object.__init__(self)
    self.factorized = None



#
# dense matrix class
#

class dense_matrix(base_matrix):
  def __init__(self, m):
    base_matrix.__init__(self, m)

    m = array(m)
    self.A     = m
    self.shape = m.shape


  def factorize(self):
    if self.factorized is not None:
      return self.factorized

    lu, piv = lu_factor(self.A)

    def factorized(b):
      return lu_solve((lu, piv), b)

    self.factorized = factorized
    return self.factorized


  def todense(self):
    return self.A



#
# sparse matrix class
#

class sparse_matrix(base_matrix):
  def __init__(self, m):
    base_matrix.__init__(self, m)

    m = csc_matrix(m)
    self.A     = m
    self.shape = m.shape


  def factorize(self):
    if self.factorized is not None:
      return self.factorized

    self.factorized = factorized(self.A)
    return self.factorized


  def todense(self):
    return self.A.todense()



#
# augmented matrix class
#

class augmented_matrix(base_matrix):
  def __init__(self, A, B, C, D):
    """ builds the matrix in the shape of

          A B
          C D

        where A         is a NxN matrix,
              D         is a scalar, and
              B and C   are of corresponding sizes. """

    base_matrix.__init__(self, A, B, C, D)

    self.A     =       A
    self.B     = array(B)
    self.C     = array(C)
    self.D     =       D
    self.shape = (A.shape[0]+1, A.shape[1]+1)


  def factorize(self):
    if self.factorized is not None:
      return self.factorized

    A, B, C, D = self.A, self.B, self.C, self.D
    Afacd = A.factorize()

    AmB = Afacd(B) # we need a solve here

    # Shur complement of A
    SC = 1.0/(D - dot(C, AmB))

    def factorized(b):
      x = b[:-1]
      y = b[-1:]
      c = zeros_like(b)

      AfacdX = Afacd(x)
      CAfacdX = dot(C, AfacdX)

      Px = AfacdX + AmB*SC*CAfacdX
      Qy =-AmB*SC*y
      Rx =-SC * CAfacdX
      Sy = SC*y

      c[:-1] = Px + Qy
      c[-1:] = Rx + Sy

      return c

    self.factorized = factorized
    return factorized


  def todense(self):
    """ returns dense representation """
    m, n = self.A.shape
    d = zeros((m+1, n+1))
    d[:-1,:-1] = self.A.todense()
    d[:-1, -1] = self.B
    d[ -1,:-1] = self.C
    d[ -1, -1] = self.D
    return d


  def dfactorize(self):
    """ dense factorize """
    lu, piv = lu_factor(self.todense())

    def factorized(b):
      return lu_solve((lu, piv), b)

    return factorized
