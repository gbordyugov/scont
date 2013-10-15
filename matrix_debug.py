from numpy import eye, ones
from numpy.linalg import norm
from numpy.random import rand, seed
from scipy.sparse import rand as sparse_rand, diags

from matrix import dense_matrix, sparse_matrix, augmented_matrix

seed()

def dense_test(core_size, aug_size, ntests):
  for unimportant_counter in xrange(ntests):
    core = rand(core_size, core_size)
    c = dense_matrix(core)

    for i in xrange(aug_size):
      d = augmented_matrix(c, rand(core_size+i),
                              rand(core_size+i), 1)
      c = d

    dfac       = d.factorize()
    dfac_dense = d.dfactorize()

    b = rand(core_size+aug_size)

    print norm(dfac(b)-dfac_dense(b))


def sparse_test(core_size, aug_size, ntests):
  for unimportant_counter in xrange(ntests):
    d0, d1, d2 = rand(core_size), rand(core_size-1), rand(core_size-1)
    core = diags([d0, d1, d2], [0, -1, 1])
    c = sparse_matrix(core)

    for i in xrange(aug_size):
      d = augmented_matrix(c, rand(core_size+i),
                              rand(core_size+i), 1)
      c = d

    dfac       = d.factorize()
    dfac_dense = d.dfactorize()

    b = rand(core_size+aug_size)
    print 'norm of diff between direct and non-direct:',\
           norm(dfac(b)-dfac_dense(b))

  
  
