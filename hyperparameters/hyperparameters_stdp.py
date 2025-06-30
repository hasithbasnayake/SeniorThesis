import numpy as np
from lgn.DoG import *

# Model Hyperparameters 

num_input = 1568
num_output = [50, 100, 200, 400]
num_steps = 255
beta = 0.9
threshold = 20
reset_mechanism = "zero"

# Update Hyperparameters

A_plus = 5e-3
A_minus = 3.75e-3
tau = 200
mu_plus = 0.65
mu_minus = 0.05

update_params = [A_plus, A_minus, tau, mu_plus, mu_minus]

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

num_samples = 1000
num_epochs = 100


