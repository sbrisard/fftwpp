import numpy as np
import pyfftwpp as fftw

import faulthandler
faulthandler.enable()

if __name__ == "__main__":
    arr1 = np.zeros((4, 5, 6), dtype=np.float64)
    arr2 = np.zeros((4, 6, 5), dtype=np.float64)
    fftw.assert_same_shape(arr1, arr2)
