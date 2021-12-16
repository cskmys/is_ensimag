import numpy as np
from keras import backend as kbe
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dat = kbe.variable(np.random.random((4, 2)))  # mk 4 x 2 tensor with random numbers
zer_dat = kbe.zeros_like(dat)  # mk 4 x 2 tensor of zeros
print(kbe.eval(zer_dat)) # should see all zeros
