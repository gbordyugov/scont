#
# pickable class
# 

import pickle

class pickable(object):
  def save(self, fname):
    f = open(fname, 'w')
    pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    f.close()


  @staticmethod
  def load(fname):
    f = open(fname, 'r')
    s = pickle.load(f)
    f.close()
    return s
