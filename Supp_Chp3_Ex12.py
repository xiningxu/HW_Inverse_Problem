# -*- coding: utf-8 -*-
"""
Supplementary Example: 
    
    Chapter3  Example 8: Gibbs Sampler
    
    (single component) Gibbs Sampler for two-dimensional distribution

@author: XiningXu 17110180016
"""


import numpy as np
import matplotlib.pyplot as plt


def fun_pi(x1, x2):
    pi_x = -10*(x1**2-x2)**2 - (x2-1/4)**4
    return np.exp(pi_x)

def build_cdf(density):
    density_area=[0]
    [density_area.append(density[i]+density[i+1]) for i in range(density.size-1)]
    return np.array(density_area)/(2*n)
    
    
def cdf_phi_1(x2):
    x1 = np.linspace(-2,2,n+1)
    df1 = fun_pi(x1, x2)
    density_area = build_cdf(df1)
    cdf1 = np.cumsum(density_area)/np.sum(density_area)
    return x1, df1/np.sum(density_area), cdf1

def cdf_phi_2(x1):
    x2 = np.linspace(-1,2,n+1)
    df2 = fun_pi(x1, x2)
    density_area = build_cdf(df2)
    cdf2 = np.cumsum(density_area)/np.sum(density_area)
    return x2, df2/np.sum(density_area), cdf2

def simple_sampling_convert(ca, cb, phi):
    alpha = -cb + np.sqrt(cb**2+8*ca*phi)
    return alpha/(2*ca)

def find_inverse(random_number, x, df, cdf):
    sample_loc = np.min(np.where((cdf - random_number)>0))
    ca = (df[sample_loc]-df[sample_loc-1])/(x[sample_loc]-x[sample_loc-1])
    cb = 2*df[sample_loc-1]
    sample = x[sample_loc-1] + simple_sampling_convert(ca, cb, random_number - cdf[sample_loc-1] )
    return sample

if __name__ == '__main__':
    n = 500
    
    x = [1.5,-0.8]
    x_record = [np.array(x)]
    """
        Gibbs Sampling
    """
    N = 50
    for iters in range(N):
        x1, df1, cdf1 = cdf_phi_1(x[1])
        x[0] = find_inverse(np.random.uniform(), x1, df1, cdf1)
        x2, df2, cdf2 = cdf_phi_1(x[0])
        x[1] = find_inverse(np.random.uniform(), x2, df2, cdf2)
        x_record.append(np.array(x))
    
    x_record = np.array(x_record)
    
    x = np.linspace(-2,2,1000)
    y = np.linspace(-2,2,1000)
    X, Y = np.meshgrid(x, y)
    P = fun_pi(X,Y)
    fig1 = plt.figure(0)
    fig1.figsize = (6,4.5)
    plt.contourf(X, Y, P, 10, cmap = plt.cm.Blues_r)
    #plt.contour(X, Y, P, 10, colors = 'r', linewidth = 0.1)
    plt.xlim(-2,2)
    plt.ylim(-1,2)
    plt.tick_params(labelsize='15')
    plt.title("Gibbs Sampling, number=" + str(N), fontsize='18')
    plt.plot(x_record[:,0],x_record[:,1],'.', color='red', markersize = 5)
    # plt.savefig("Gibbs Sampling test.eps")
    
        
