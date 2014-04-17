from sector import sector
from finger import finger
from f2s import f2s
from ucont import ucont

f = finger.load('fingers/ic/e-0.3.finger')
s = f2s(f, 1.0e6, {'dummy':0.0})

# determine the ``right'' epsilon value
branch, s = ucont(s, 'dummy', 'e', 'omega',     1,    1.0)
branch, s = ucont(s,     'r', 'e', 'omega', 10000, -500.0)
