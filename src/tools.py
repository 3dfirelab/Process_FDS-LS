from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
import socket
import sys,os
import numpy as np 
import matplotlib as mpl 
if 'matplotlib.pyplot' not in sys.modules: mpl.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
import glob 
import shapefile
from PIL import Image, ImageDraw
import itertools
from scipy import interpolate
import pdb 
import math 
import pandas
import datetime
import multiprocessing
import subprocess

#homebrewed
sys.path.append('./Factor_number/')
import factor


################################################
def cpu_count():
    try:
        return int(os.environ['ntask'])
    except:
        print('env variable ntask is not defined')
        sys.exit()
        #return multiprocessing.cpu_count()


#####################################################
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d) 

######################################################
def downgrade_resolution(arr, diag_res_cte_shape, flag_interpolation='conservative', flag_grid=None):


    '''
    flag_interpolation is conservative, or use max value in the new grid box
    '''
    factor = old_div(1.*arr.shape[0],diag_res_cte_shape[0])
    if factor == np.int(np.floor(factor)): factor = np.int(factor)
    else: factor = np.int(factor) + 1
    
    #if np.mod( arr.shape[0], factor )!=0:
    if factor*diag_res_cte_shape[0] -arr.shape[0] != 0:
        extra_pixel0 = factor*diag_res_cte_shape[0] -arr.shape[0]

        extra_pixel0l = np.int(0.5*extra_pixel0)
        extra_pixel0r = extra_pixel0 - extra_pixel0l
    else: 
        extra_pixel0 = 0
        
    if factor*diag_res_cte_shape[1] -arr.shape[1] != 0:
        extra_pixel1 = factor*diag_res_cte_shape[1] -arr.shape[1]
        
        extra_pixel1l = np.int(0.5*extra_pixel1)
        extra_pixel1r = extra_pixel1 - extra_pixel1l
    else: 
        extra_pixel1 = 0
 
    if (extra_pixel0>0) |  (extra_pixel1>0):

        if flag_grid == 'grid_e': 
            '''
            as we avoir extrapolation, we rebuilt the grid in this case
            '''
            dx = arr[1,0]-arr[0,0] 
            xb = arr[0,0]-dx*extra_pixel0l
            xe = arr[-1,0]+dx*extra_pixel0r

            x_ = np.linspace(xb,xe,arr.shape[0]+extra_pixel0)

            arr2 = np.zeros([arr.shape[0]+extra_pixel0,arr.shape[1]+extra_pixel1])
            for jj in range(arr.shape[1]+extra_pixel1):
                arr2[:,jj] = x_
       
            arr = arr2
        
        elif flag_grid == 'grid_n': 
            '''
            as we avoir extrapolation, we rebuilt the grid in this case
            '''
            dy = arr[0,1]-arr[0,0] 
            yb = arr[0,0] -dy*extra_pixel1l
            ye = arr[0,-1]+dy*extra_pixel1r

            y_ = np.linspace(yb,ye,arr.shape[1]+extra_pixel1)

            arr2 = np.zeros([arr.shape[0]+extra_pixel0,arr.shape[1]+extra_pixel1])
            for ii in range(arr.shape[0]+extra_pixel0):
                arr2[ii,:] = y_
            arr = arr2
       

        else:
            x = np.arange(0,arr.shape[0],1)
            y = np.arange(0,arr.shape[1],1)
            z = arr.flatten()
            f = interpolate.interp2d(x, y, z, kind='linear')
            
            grid_x = np.arange(0-extra_pixel0l,extra_pixel0r+arr.shape[0],1)
            grid_y = np.arange(0-extra_pixel1l,extra_pixel1r+arr.shape[1],1) 
            arr2 = f(grid_x, grid_y)
            arr2 = arr2.T
            #pdb.set_trace()
            arr = arr2

    if flag_interpolation == 'max':
        outarr = shrink_max(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
   
    elif flag_interpolation == 'min':
        outarr = shrink_min(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
    
    elif flag_interpolation == 'conservative':
        
        mask = np.where(arr!=-999, 1, 0)
        sum_pixel = shrink_sum(mask, diag_res_cte_shape[0], diag_res_cte_shape[1]) 
        
        sum = shrink_sum(arr, diag_res_cte_shape[0], diag_res_cte_shape[1]) 
        outarr = np.where(sum != -999, old_div(sum,sum_pixel), sum)

    elif flag_interpolation == 'average':
        outarr = shrink_average(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
    
    elif flag_interpolation == 'sum':
        outarr = shrink_sum(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])

    else:
        print('bad flag')
        pdb.set_trace()


    return outarr

