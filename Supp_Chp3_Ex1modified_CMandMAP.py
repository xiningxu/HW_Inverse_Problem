# -*- coding: utf-8 -*-
"""
Suppplementary Example:
        Chapter3 Example1(modified): 
            A comparision between CM estimate and MAP estimate

@author: Xining Xu 17110180016
"""

import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt


def phi(x):
    y = np.exp(-(x**2)/2)
    return y/np.sqrt(2*np.pi)

def test_density(x,a, std1, std2):
    pi_1 = a*phi(x/std1)/std1
    pi_2 = (1-a)*phi((x-1)/std2)/std2
    return pi_1+pi_2

def simple_sampling_convert(ca, cb, phi):
    alpha = -cb + np.sqrt(cb**2+8*ca*phi)
    return alpha/(2*ca)

def simple_sampling(x, a, std0, std1, sample_max):
    df = np.array(test_density(x,a,std0,std1))
       
    density_area=[0]
    [density_area.append(df[i]+df[i+1]) for i in range(df.size-1)]
    density_area = (x[1]-x[0])*np.array(density_area)/2
    
    cdf = np.cumsum(density_area)/np.sum(density_area)
    df = df/np.sum(density_area)
    sample_record = []
    for i in range(sample_max):
        rnd_sample = np.random.uniform()
        sample_loc = np.min(np.where((cdf-rnd_sample)>0))
        ca = (df[sample_loc]-df[sample_loc-1])/(x[sample_loc]-x[sample_loc-1])
        cb = 2*df[sample_loc-1]
        sample_record.append(x[sample_loc-1] +
            simple_sampling_convert(ca, cb, rnd_sample-cdf[sample_loc-1]))
    
    return np.array(sample_record)
    

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    """
        Settings
    """  
    n = 10000
    
    x1 = -0.5
    x2 = 1.5
    x = np.linspace(x1, x2, n+1)
    
    sample_max = 5000
    cont = 100
    figcont = -1
    """
      Simple sampling
    """
    a_l = [0.5, 0.01]
    std0_l = [0.08, 0.001]
    std1_l = [0.04, 0.1]
    
    for i in range(2):
        a = a_l[i]
        std0 = std0_l[i]
        std1 = std1_l[i]
        sample = simple_sampling(x, a, std0, std1, sample_max)    
        figcont += 1
        plt.figure(figcont,figsize=(6,4))
        sns.distplot(sample, bins = 400, rug = False, kde = False,
                     hist = True, norm_hist = True ,
                     hist_kws = {'histtype':'stepfilled', 'linewidth':1,
                                 'alpha':1, 'color':'silver'}) 
        plt.plot(x,np.array(test_density(x,a,std0,std1)),
                 color = 'darkred', linewidth = '3',linestyle = '--',
                 label = r"posterior pdf: $\pi(x)$")
        plt.legend(fontsize = 15, loc = "upper left")
        plt.xticks(np.linspace(-0.5,1.5,5))
        plt.tick_params(labelsize='15')
        plt.title("Simple Sampling")
        plt.xlabel("$x$",fontsize = '14')
        plt.xlim(-0.5,1.5)
        plt.ylim(0,6)
        #plt.savefig("chap3_ex1_2.eps")
        
       
        sample_mu = []
        sample_std = []
        sample_map =[]
        for i  in range(cont):
            sample = simple_sampling(x, a, std0, std1, sample_max) 
            sample_mu.append(np.mean(sample))
            sample_std.append(np.std(sample))
            sample_val = test_density(sample,a, std0, std1)
            map_loc = np.where(sample_val == np.max(sample_val))
            sample_map.append(float(sample[map_loc]))
            
            if i%100 == 0:
                print(str(i))
                
        sample_mu = np.array(sample_mu)  
        sample_std = np.array(sample_std)
        sample_map = np.array(sample_map)
        cm_mu = 1-a
        cm_std = a*std0**2 + (1-a)*(std1**2 + 1) - (1-a)**2
        cm_std = np.sqrt(cm_std)
        map_est = 0 + 1*((a/std0)<((1-a)/std1))
        
        figcont += 1
        plt.figure(figcont, figsize=(6,4))
        plt.plot(np.arange(cont)+1, (cm_mu + cm_std)*np.ones((cont)),'-',
                 linewidth = 3, color = 'hotpink')
        plt.plot(np.arange(cont)+1, (cm_mu - cm_std)*np.ones((cont)),'-',
                 linewidth = 3, color = 'hotpink')
        
        plt.plot(np.arange(cont)+1, cm_mu*np.ones((cont)), '-', 
                 color = "navy", linewidth = 4, label = "CM estimate")
        plt.plot(np.arange(cont)+1, sample_mu, '--', 
                 color = "red", linewidth = 4, label = "CM estimate(sample)")
        
    
        plt.fill_between(np.arange(cont)+1, sample_mu + sample_std,
                         sample_mu - sample_std, color = 'silver')
        
        plt.plot(np.arange(cont)+1, map_est*np.ones((cont)),':',
                 linewidth = 4, color = 'blue', label = "MAP estimate")
        plt.ylim(-0.5,2.5)
        plt.xlim(1,cont)
        plt.yticks(np.linspace(-0.5,2.5,7))
        tk = np.linspace(0,cont,6)
        tk[0] = 1
        plt.xticks(tk)
        plt.tick_params(labelsize='15')
        plt.legend(fontsize = "12")
        #plt.savefig("chap3_ex1_4.eps")
        print(str(i))
    
        
