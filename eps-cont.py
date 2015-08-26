from sector import sector
from finger import finger
from f2s import f2s
from ucont import ucont

f = finger.load('fingers/ic/e-0.2.finger')

def z1(x, p):
  return p - 0.1

branch, f = ucont(f, 'e', 'vy', 'b',  1000, -1.0, [z1])
f.save('fingers/ic/e-0.1.finger')
