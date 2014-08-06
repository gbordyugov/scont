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

f = finger.load('fingers/ic/e-0.2.finger')
f.reshape(300,200)
def z1(x, p): return p - 0.1
branch, f = ucont(f, 'e', 'vy', 'b',  1000, -1.0, [z1])
f.save('fingers/ic/e-0.1.finger')