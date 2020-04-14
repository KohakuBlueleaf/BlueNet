try:
	import cupy as _np
	import cupyx.scipy.special.erf as _erf
	from cupy import exp as _exp
except:
	import numpy as _np
	from scipy.special import erf as _erf
	from numpy import exp as _exp