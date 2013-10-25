class dview(dict):
  pass
  """ This class provides a view to an external dictionary. The view is
      limited by the set of given keys."""
  def __init__(self, d, keys=[]):
    self.d    = d
    self.keys = keys

  def __getitem__(self, key):
    if key in self.keys:
      return self.d[key]
    else:
      raise KeyError('dview.__getitem__(): key %s is not in the list of keys' % (str(key)) )

  def __setitem__(self, key, value):
    if key in self.keys:
      self.d[key] = value
    else:
      raise KeyError('dview.__setitem__(): key %s is not in the list of keys' % (str(key)) )

  def get_dict(self):
    d = {}

    for k in self.keys:
      if k in self.d:
        d[k] = self.d[k]

    return d


  def __repr__(self):
    return repr(self.get_dict())

  def __str__(self):
    return  str(self.get_dict())
