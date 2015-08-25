from sector import sector
from finger import finger
from f2s import f2s
from ucont import ucont


# f.nx = 225, f.ny = 150
f = finger.load('fingers/nsp-3.finger')

cutidx = 170
old_data = f.data[:, cutidx:, :]
old_ly = f.ly
old_ny = f.ny
f.data = old_data.copy()
f.ny = f.ny - cutidx
f.ly = old_ly*f.ny/old_ny
f.nsp = 3
# f.reshape(150, 300)
branch, f = ucont(f, 'dummy', 'vy', 'b',  1, 1.0)
# f.save('fingers/fc.finger')
