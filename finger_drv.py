from numpy import loadtxt
from finger import finger
from fhn import FHNNonlinearity, FHNJacobian

pars = {'nx'     : 225,     # those ones are to be matched to those
        'ny'     : 150,     # of the file 'fingers/25500.dat'
        'nv'     : 2,
        'lx'     : 75.0,              # x length
        'ly'     : 58.17764173314432, # y length
        'vx'     : 0.0,               # x velocity
        'vy'     : 1.450126183373418, # y velocity
        'a'      : 0.5,
        'b'      : 0.874238328501226,
        'e'      : 0.29993351474573732,
        'f'      : FHNNonlinearity,    # nonlinearity
        'jac'    : FHNJacobian,        # jacobian
        'D'      : [[1.0, 0.0],
                    [0.0, 0.0]]}


f = finger(pars)
f.flat[:] = loadtxt('fingers/25500.dat')

print f.lapmatrix().shape
f.save('fingers/start.finger')
