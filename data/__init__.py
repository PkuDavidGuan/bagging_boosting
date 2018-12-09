from .Seismic import Seismic
from .Cnae import Cnae

__factory = {
  'seismic': Seismic,
  'cnae': Cnae,
  'airfoil': Airfoil,
  'concrete': Concrete,
}


def create(name, root, *args, **kwargs):
  """
  Create a dataset instance.
  Parameters
  ----------
  name : str
    The dataset name. Can be one of __factory
  root : str
    The path to the dataset directory.
  """
  if name not in __factory:
    raise KeyError("Unknown dataset:", name)
  return __factory[name](root, *args, **kwargs)