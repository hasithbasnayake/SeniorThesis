import numpy as np
from lgn.DoG import *

# Model Hyperparameters 

num_input = 1568
num_hidden = [50]
num_output = 10
num_steps = 255
beta = 0.9
threshold = 1.0
reset_mechanism = "subtract"

# Learning Hyperparameters

learning_rate = 5e-4

# Kernel Params

# Kernel Hyperparameters

dim = np.array([6, 6]) 
ppa = np.array([8, 8])
ang = np.ceil(dim / ppa)
ctr = (1/3) * dim[0]
sur = (2/3) * dim[0]

ON_kernel = DOG_kernel(dim, ang, ppa, ctr, sur)
OFF_kernel = DOG_kernel(dim, ang, ppa, ctr, sur)

ON_kernel.set_filter_coefficients(ONOFF="ON")
OFF_kernel.set_filter_coefficients(ONOFF="OFF")

dog_transform = DualDoG(ON_kernel, OFF_kernel)

# Training Hyperparameters

batch_size = 128
num_samples = 1000
num_epochs = 10


