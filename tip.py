from scipy.interpolate  import  interp2d
from scipy.optimize     import  root, fsolve
from numpy import zeros_like, meshgrid, zeros

def ComputeContour(u, threshold=None):
  """ Computes contour of u given the threshold """

  if threshold == None:
    threshold = (u.max() + u.min())/2.0

  m = zeros_like(u, dtype='int32') # mask
  d = zeros_like(u, dtype='int32') # derivative of the mask
  m[u >= threshold] = 1
  d[:-1,:-1] = 3*m[:-1,:-1] - m[0:-1,1:] - m[1:,0:-1] - m[1:,1:]
  return d != 0


def FindTip(u, v, ut=None, vt=None):
  """ Finds the tip of the spiral as the section of u and v contour
      lines. Contours are given by the thresholds ut, vt. """

  if ut is None:
    ut = (u.max() + u.min())/2.0

  if vt is None:
    vt = (v.max() + v.min())/2.0

  if u.shape != v.shape:
    print 'different shapes of u and v in FindTip'
    return

  (nx, ny) = u.shape

  J, I = meshgrid(range(ny), range(nx)) # meshgrid assumes x-wise storage

  m = ComputeContour(u, ut) & ComputeContour(v, vt)

  foundTips = {}

  for i, j in zip(I[m], J[m]):
    if (i-3 < 0 or i+3 > nx) or (j-3 < 0 or j+3 > ny):
      continue

    i1, i2 = i-3, i+3
    j1, j2 = j-3, j+3

    u_sub = u[i1:i2, j1:j2].transpose() # interp2d assumes x-wise storage
    v_sub = v[i1:i2, j1:j2].transpose() # interp2d assumes x-wise storage
    U = interp2d(range(i1, i2), range(j1, j2), u_sub, kind='quintic')
    V = interp2d(range(i1, i2), range(j1, j2), v_sub, kind='quintic')

    def f(p):
      x, y = p[0], p[1]
      return [U(x, y)[0] - ut, V(x, y)[0] - vt]

    x = root(f, [i+0.5, j+0.5]).x
    foundTips[(int(x[0]), int(x[1]))] = x.tolist()

  return foundTips.values()

   
def DTipDS (data, ut=0.0, vt=0.0, h=1.0e-5):
  """
  Computes the derivative of the tip position with respect to
  the components of the RDS. Seems to be useless so far.
  """
  data = data.copy()
  (nx, ny, nv) = data.shape
  u = data[...,0]
  v = data[...,1]

  t = FindTip(u, v, ut, vt)[0]
  ti, tj = int(t[0]), int(t[1])

  if (ti-5 < 0 or ti+5 > nx) or (tj-5 < 0 or tj+5 > ny):
    print 'tip too close to boundary'
    return None, None

  dtxds = zeros(nx*ny*nv)
  dtyds = zeros(nx*ny*nv)

  dtxds3 = dtxds.reshape((nx, ny, nv))
  dtyds3 = dtyds.reshape((nx, ny, nv))

  for i in xrange(ti-5, ti+5):
    for j in xrange(tj-5, tj+5):
      for k in xrange(nv):
        data[i,j,k] += h/2.0
        t2 = FindTip(u, v, ut, vt)[0]

        data[i,j,k] -= h/2.0
        t1 = FindTip(u, v, ut, vt)[0]

        dtxds3[i,j,k] = (t2[0] - t1[0])/h
        dtyds3[i,j,k] = (t2[1] - t1[1])/h

        data[i,j,k] += h/2.0


  return dtxds, dtyds
















