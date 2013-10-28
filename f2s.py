#!/usr/bin/python

from finger import finger
from sector import sector

from sys import argv, exit

def f2s(f, r, d = {}):
  pars = {'nr'        : f.nx,
          'ntheta'    : f.ny,
          'nv'        : f.nv,
          'nsp'       : f.nsp,
          'r'         : r,
          'R'         : f.lx,
          'arclength' : f.ly,
          'omega'     : f.vy/r,
          'a'         : f.a,
          'b'         : f.b,
          'e'         : f.e,
          'f'         : f.f,
          'jac'       : f.jac,
          'D'         : f.D}
  
  pars.update(d)
  s = sector(pars)
  s.flat[:] = f.flat[:]
  return s


if __name__ == '__main__':
  if len(argv) < 4:
    print 'usage: f2s.py name.finger name.sector radius'
    exit(-1)
  
  fname =       argv[1]
  sname =       argv[2]
  r     = float(argv[3])
  
  print "converting %s to %s with radius %f" % (fname, sname, r)
  
  f = finger.load(fname)
  
  print 'finger parameters:'
  print f.pars
  
  s = f2s(f, r)
  
  print 'sector parameters:'
  print s.pars
  
  s.save(sname)
