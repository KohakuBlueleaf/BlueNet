import numpy as _np
from scipy.special import erf as _erf
from numpy import exp as _exp
device = 'CPU'

def change_device(mode):
  global _np, _erf, _exp,device
  if mode.upper()=='GPU':
    try:
      import cupy as _np
      try:
        import cupyx.scipy.special.erf as _erf
      except:
        from cupyx.scipy.special import erf as _erf
      from cupy import exp as _exp
      _np.array([1,1,1])
      device = 'Cuda'
    except Exception as e:
      print(e)
      import numpy as _np
      from scipy.special import erf as _erf
      from numpy import exp as _exp
      device = 'CPU'
  else:
    import numpy as _np
    from scipy.special import erf as _erf
    from numpy import exp as _exp
    device = 'CPU'
