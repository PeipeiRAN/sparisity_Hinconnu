""" SPARSITY

Functions specific to sparsity regularisation.

"""

import numpy as np
from scipy.linalg import svd, diagsvd
from scipy.linalg import norm
from shared_functions import *
from datetime import datetime
from astropy.io import fits
from os import remove
from subprocess import check_call
#import os

#os.chdir('/Users/pran/Documents/code/simplescriptexample')
#%%
def call_mr_transform(data, opt=None, path='./', remove_files=True):
    
    #this method calls the iSAP module mr_transform
    
    # Create a unique string using the current date and time.
    unique_string = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    
    # Set the ouput file names.
    file_name = path + 'mr_temp_' + unique_string
    file_fits = file_name + '.fits'
    file_mr = file_name + '.mr'

    # Write the input data to a fits file.
    fits.writeto(file_fits, data)
    
    # Call mr_transform.
    if isinstance(opt, type(None)):
        check_call(['mr_transform', file_fits, file_mr])
    else:
        check_call(['mr_transform'] + opt + [file_fits, file_mr])

    # Retrieve wavelet transformed data.
    result = fits.getdata(file_mr)

    # Return the mr_transform results (and the output file names).
    if remove_files:
        remove(file_fits)
        remove(file_mr)
        return result
    else:
        return result, file_mr  
#%%
    
def get_mr_filters(data_shape, opt=None, coarse= False):
    
    #adjust the shape of the input data
    data_shape = np.array(data_shape)
    data_shape += data_shape%2 - 1
    
    #create fake data
    fake_data = np.zeros(data_shape)
    fake_data[zip(data_shape/2)] = 1
    
    #get the mr_filter
    mr_filters = call_mr_transform(fake_data, opt=opt)
    
    if coarse:
       return mr_filters
    else:
       return mr_filters[:-1]
#%%
class cwbReweight(object):
    
    def __init__(self, weights, thresh_factor=1):

        self.weights = weights
        self.original_weights = np.copy(self.weights)
        self.thresh_factor = thresh_factor

    def reweight(self, data):
        
        """
        Reweighting implemented as w = w (1 / (1 + |x^w|/(n * sigma)))

        """

        self.weights *= (1.0 / (1.0 + np.abs(data) / (self.thresh_factor *
                         self.original_weights)))
        return self.weights
        
#%%

def get_weight(data,psf):
    
    noise_est = sigma_mad(data)
    wave_thresh_factor = [3.0, 3.0, 4.0]
    data_shape = data.shape[-2:] #[41,41]
    
    mr_filters = get_mr_filters(data_shape, opt=None, coarse= False)#[3,41,41]
    
    filter_conv = (filter_convolve_stack(np.rot90(psf, 2),
                       mr_filters))#[100,3,41,41]
    filter_norm = np.array([[norm(b) * c * np.ones(data_shape[1:])
                                for b, c in zip(a, wave_thresh_factor)]
                                for a in filter_conv])

   
    weight0 = noise_est * filter_conv
    weight = cwbReweight(weight0)

    return weight.weights

def denoise(data, level, threshold_type='hard'):
    
    if threshold_type not in ('hard', 'soft'):
        raise ValueError('Invalid threshold type. Options are "hard" or'
                         '"soft"')

    if threshold_type == 'soft':
        return np.sign(data) * (np.abs(data) - level) * (np.abs(data) >= level)

    else:
        return data * (np.abs(data) >= level)
    
    

class Threshold(object):
    """
    Threshold proximity operator for update the weight and apply the proximity

    """

    def __init__(self, weights):

        self.update_weights(weights)

    def update_weights(self, weights):
       

        self.weights = weights

    def op(self, data, extra_factor=1.0):

        threshold = self.weights * extra_factor

        return denoise(data, threshold, 'soft')

#%%
def prox_dual_op_s(a, thresh):
    
    return soft(a, thresh)

#def prox_dual_op_s(a, thresh):

#    return Threshold(a, thresh).op
    
def linear_op(data):
    
    data_shape = data.shape[-2:]
    mr_filters = get_mr_filters(data_shape, opt=None, coarse= False)
    linear_data = filter_convolve_stack(data, mr_filters)
    
    return linear_data

def linear_op_inv(data):
     
    data_shape = data.shape[-2:]
    mr_filters = get_mr_filters(data_shape, opt=None, coarse= False)
    linear_data_inv = filter_convolve_stack(data, mr_filters, filter_rot=True)
    
    return linear_data_inv

def get_cost(x, data, psf, thresh, psf0):
    
    data_shape = data.shape[-2:]
    mr_filters = get_mr_filters(data_shape, opt=None, coarse= False)
    l1_norm = thresh*filter_convolve_stack(x, mr_filters)
    l1_norm = np.sum(np.abs(l1_norm))
   
    #df = ((convolve_stack(x, psf) - data)**2).sum()
    
    df = np.linalg.norm((convolve_stack(x, psf) - data)**2)
    
    l2_norm_psf = 0.01 * np.linalg.norm(psf - psf0) 
    
    #df = np.array([np.linalg.norm(d)**2 for d in (convolve_stack(x, psf) - data)])
   
    #df = np.sum(df)
    
    #l2_norm_psf = np.array([np.linalg.norm(p) for p in (psf - psf0)])

    #l2_norm_psf = np.sum(l2_norm_psf)
    
    print "get_cost:"
    print ''
    
    return (0.5 * df + l1_norm + l2_norm_psf)

