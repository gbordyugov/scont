from sector import sector
from finger import finger
from f2s import f2s
from ucont import ucont

par2ix = -2
par3ix = -1
par1ix =  0

# f = finger.load('fingers/ic/e-0.3.finger')
# 
# def z0(x, p): return p - 0.2
# branch, f = ucont(f, 'e', 'vy', 'b',  1000, -5.0, [z0])
# f.save('fingers/ic/e-0.2.finger')

# f.nx = 225, f.ny = 150
f = finger.load('fingers/nsp-3.finger')
f.reshape(225, 150)
# f.reshape(225, 150)

branch, f = ucont(f, 'dummy', 'vy', 'b',  1, 1.0)
f.save('fingers/fc.finger')
