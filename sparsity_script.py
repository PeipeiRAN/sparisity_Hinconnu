import sys
import numpy as np
import matplotlib.pyplot as plt
from shared_functions import *
from sparsity_functions import *
from shape import *

import os
os.chdir('/Users/pran/Documents/code/new')

# INPUT DATA
#data = np.load('example_image_stack.npy')
#psf = np.load('example_psfs.npy')

data = np.load('galaxies_obs.npy')
psf = np.load('PSF_interp.npy')
data_true = np.load('galaxies_tru.npy')
psf_true = np.load('PSF_tru.npy')
psf0 = psf
psf_rot = rotate_stack(psf)

# SETUP
n_iter = 100
init_cost = 1e6
tolerance = 1e-5
n_iter_reweight = 25 #reweight number = 35, the result is better than Sam's, the whole iteration is like 270

# CONDAT
condat_sigma = condat_tau = 0.5
condat_relax = 0.5

# SPARSE
sparse_thresh = get_weight(data,psf) #[100,3,41]
#%%
print 'rho:', condat_relax
print 'sigma:', condat_sigma
print 'tau:', condat_tau
print ''
#print 'threshold:', sparse_thresh
nmse_psf0 = nmse(psf_true, psf, metric=np.mean)
print 'nmse_psf0:', nmse_psf0

nmse_data0 = nmse(data_true, data, metric=np.mean)
print 'nmse_data0:', nmse_data0

e_error_data0 = e_error(data, data_true, metric=np.mean)
print 'e_error_data0:', e_error_data0
#%%

# PROXIMIAL VARIABLES
x = np.copy(data)

data_shape = data.shape[-2:]
mr_filters = get_mr_filters(data_shape, opt=None, coarse= False)
dual_shape = [mr_filters.shape[0]]+ list(data.shape)
dual_shape[0], dual_shape[1] = dual_shape[1], dual_shape[0]
dual_shape = dual_shape
y = np.ones(dual_shape)
print dual_shape
#%%
# OPTIMISATION
costs = []
nmse_psfs = []
e_error_datas = []
lamda = 1
for iter_num_reweight in xrange(0, n_iter_reweight):
    
    print '-REWEIGHT:', iter_num_reweight + 1
    print ' '

    for iter_num in xrange(1, n_iter):
        
       x_rot = rotate_stack(x)
       
       #lamda = get_lamda(lamda)
       
       psf = get_grad_psf(x, data, psf, psf0, x_rot, lamda) 
       
       psf_rot = rotate_stack(psf) 
       
       grad = get_grad(x, data, psf, psf_rot)

       x_prox = prox_op(x - condat_tau * grad - condat_tau * linear_op_inv(y))

       y_temp = (y + condat_sigma *linear_op((2 * x_prox) - x))
#%%
       y_prox = (y_temp - condat_sigma *
              prox_dual_op_s(y_temp / condat_sigma, sparse_thresh / condat_sigma))

       x = condat_relax * x_prox + (1 - condat_relax) * x
       y = condat_relax * y_prox + (1 - condat_relax) * y
       tmp = get_cost(x, data, psf, sparse_thresh, psf0)
       costs.append(tmp)
       print iter_num, 'COST:', costs[-1]
       print ''
       
       nmse_psf = nmse(psf_true, psf, metric=np.mean)
       nmse_psfs.append(nmse_psf)
       #print iter_num, 'nmse_psf:', nmse_psfs[-1]
       #print ''
       
       #e_error_data = e_error(data_true, x, metric=np.mean)
       #e_error_datas.append(e_error_data)
       #print iter_num, 'e_error_data:', e_error_datas[-1]
       #print ''

       if not iter_num % 4:
          cost_diff = np.linalg.norm(np.mean(costs[-4:-2]) - np.mean(costs[-2:]))
          print ' - COST DIFF:', cost_diff
          print ''

          if cost_diff < tolerance:
               print 'Converged!'
               
               print 'nmse_psf0:', nmse_psf0
               print ''
               print 'nmse_data0:', nmse_data0
               print ''
               print 'e_error_data0:', e_error_data0
               print ''
               
               nmse_data = nmse(data_true, x, metric=np.mean)
               print 'nmse_data', nmse_data
               print ''
               
               e_error_data = e_error(data_true, x, metric=np.mean)
               print 'e_error_data:', e_error_data
               print ''
               
               break
       
    sparse_thresh = get_weight(x,psf)

# OUTPUT DATA

# DISPLAY SOME IMAGES
plt.figure(1)
plt.subplot(221)
plt.imshow(x[8], interpolation='nearest')
plt.subplot(222)
plt.imshow(x[18], interpolation='nearest')
plt.subplot(223)
plt.imshow(x[28], interpolation='nearest')
plt.subplot(224)
plt.imshow(x[38], interpolation='nearest')
plt.show()

# DISPLAY COST FUNCTION DECAY
plt.figure(2)
plt.plot(range(len(costs)), costs, 'r-')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()

plt.figure(3)
plt.plot(range(len(nmse_psfs)), nmse_psfs, 'r-')
plt.xlabel('Iteration')
plt.ylabel('nmse_psfs')
plt.show()

#plt.figure(4)
#plt.plot(range(len(e_error_datas)), e_error_datas, 'r-')
#plt.xlabel('Iteration')
#plt.ylabel('e_error_datas')
#plt.show()

