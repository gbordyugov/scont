from sector import sector
from fhn import FHNNonlinearity, FHNJacobian

consts = {'nr'     : 225,     # those ones are to be matched to those
          'ntheta' : 150,     # 'sectors/150.sector'
          'nv'     : 2}

pars = {'r'      : 150.0,              # inner radius of the ring
        'R'      : 75.0,               # thickness of the ring
        'theta'  : 0.3878509448876288, # opening angle of the sector
        'omega'  : 0.0079106968529917675,
        'a'      : 0.5,
        'b'      : 0.874238328501226,
        'e'      : 0.29383566966615743,
        'f'      : FHNNonlinearity,    # nonlinearity
        'jac'    : FHNJacobian,        # jacobian
        'D'      : [[1.0, 0.0],
                    [0.0, 0.0]]}

from numpy import loadtxt
pars.update(consts)
s = sector(pars)
s.flat[:] = loadtxt('/home/bordyugov/src/finger/sctrs/150.dat')

print s.lapmatrix().shape
