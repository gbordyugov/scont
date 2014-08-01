from sector import sector
from finger import finger
from f2s import f2s
from ucont import ucont

par2ix = -2
par3ix = -1
par1ix =  0

f = finger.load('fingers/ic/e-0.3.finger')
s = f2s(f, 1.0e6, {'dummy':0.0})

# determine the ``right'' epsilon value

branch, s = ucont(s, 'dummy', 'e', 'omega',     1,    1.0)

def z0(x, p):
  return x[par3ix] - 1.5e-6

branch, s = ucont(s,     'r', 'e', 'omega', 10000, -500.0, [z0])
