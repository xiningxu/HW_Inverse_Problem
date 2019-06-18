# -*- coding: utf-8 -*-
"""
Chapter 4: 
            Example 4.3: SIR(sampling importance resampling)

@author: XiningXu 17110180016
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def rayleigh_fun(x):
    x = np.maximum(x,0)
    return x*np.exp(-x**2/2)

def int_density(x,y):
    z = x**2 + (y/gamma)**2
    return x*np.exp(-z/2)/(gamma*np.sqrt(2*np.pi))

def prediction_density(y, gamma):
    y = np.matrix(y)
    t = np.matrix(np.linspace(0,10,501)).T
    dt = float(t[1]-t[0])
    
    A = int_density(np.array(t+np.zeros(y.size)), np.array(t-y))
    density = np.sum(A, axis=0)
    density = density - 0.5*A[0] - 0.5*A[-1]
    density = density * dt
    return density

def gauss(x,std):
    return np.exp(-(x/std)**2/2)

def likelihood_density(x,y,sigma):
    return gauss(y-x,sigma)/(sigma*np.sqrt(2*np.pi))


if __name__=='__main__':
    warnings.filterwarnings('ignore')
    sigma = 0.1
    gamma = 0.8
    N = 1000
    figcont = 0
    x0 = np.linspace(-3.5,8,N)
    
    x1 = np.random.rayleigh(scale = 1.0, size = (N,1))
    w1 = np.random.normal(loc = 0, scale=gamma, size = (N,1))
    x1_u = np.array(x1 + w1)
    
    v = np.random.normal(loc = 0, scale = sigma, size = (N,1))
    y = np.array(x1_u + v)
    z = y-x1_u
    w_u = gauss(z, sigma)
    w_u = w_u/np.sum(w_u)
    W = np.cumsum(w_u)
    
 
    x_k =[]
    for i in range(N):
        t = np.random.uniform(0,1,1)
        l = np.min(np.where((W-t)>=0))
        x_k.append(float(x1_u[l]))    
    x_k = np.array(x_k)
    
    
    figcont += 1
    plt.figure(figcont)
    sns.distplot(x1, bins = 50, rug = False, kde = False,
                 hist = True, norm_hist = True ,
                 hist_kws = {'histtype':'stepfilled', 'linewidth':1,
                             'alpha':1, 'color':'silver'}) 
    plt.plot(x0,rayleigh_fun(x0),
             color = 'darkred', linewidth = '2.5',linestyle = '-')
    plt.plot(x0,prediction_density(x0,gamma),
             color = 'darkblue', linewidth = '3',linestyle = '--')

    plt.xlim(-3.5,8)
    plt.ylim(0,0.8)
    plt.title(r"$k=0$",fontsize = 20)
    plt.legend([r"$\pi(x_k\mid D_k)$", r"$\pi(x_{k+1}\mid D_k)$"],
                fontsize = 15)
    plt.tick_params(labelsize=15)
    plt.savefig("chap4_ex3_1.eps")
    
    figcont += 1
    plt.figure(figcont)
    sns.distplot(x1_u, bins = 50, rug = False, kde = False,
                 hist = True, norm_hist = True ,
                 hist_kws = {'histtype':'stepfilled', 'linewidth':1,
                             'alpha':1, 'color':'silver'}) 
    plt.plot(x0,prediction_density(x0,gamma),
             color = 'darkred', linewidth = '3',linestyle = '-')
    plt.plot(x0,rayleigh_fun(x0),
             color = 'darkblue', linewidth = '2.5',linestyle = '--')
    plt.xlim(-3.5,8)
    plt.ylim(0,0.8)
    plt.title(r"$k=0$",fontsize = 20)
    plt.legend([r"$\pi(x_{k+1}\mid D_k)$", r"$\pi(x_k\mid D_k)$"],
                fontsize = 15)
    plt.tick_params(labelsize=15)
    plt.savefig("chap4_ex3_2.eps")
    
    figcont += 1
    plt.figure(figcont)
    sns.distplot(x_k, bins = 50, rug = False, kde = False,
                 hist = True, norm_hist = True ,
                 hist_kws = {'histtype':'stepfilled', 'linewidth':1,
                             'alpha':1, 'color':'silver'}) 
    plt.plot(x0,prediction_density(x0,gamma),
             color = 'darkblue', linewidth = '2.5',linestyle = '--')
    plt.plot(x0+np.mean(x_k),prediction_density(x0,gamma),
             color = 'darkred', linewidth = '3',linestyle = '-')
    
    plt.xlim(-3.5,8)
    plt.ylim(0,0.8)
    plt.title(r"$k=0$",fontsize = 20)
    plt.legend( [r"$\pi(x_{k+1}\mid D_k)$",r"$\pi(y_{k+1}\mid \bar{x}_{k+1})$"],
                fontsize = 15)
    plt.tick_params(labelsize=15)
    plt.savefig("chap4_ex3_3.eps")


    last_figure_y = prediction_density(x0+np.mean(x_k),gamma)*prediction_density(x0,gamma)
    last_figure_y = last_figure_y / (sum(last_figure_y)*(x0[1]-x0[0]))
    W_y = np.cumsum(last_figure_y)/np.sum(last_figure_y)
    
    x_k_1 =[]
    for i in range(N):
        t = np.random.uniform(0,1,1)
        l = np.min(np.where((W_y-t)>=0))
        x_k_1.append(float(x0[l])+np.mean(x_k))    
    x_k_1 = np.array(x_k_1)
    
    figcont += 1
    plt.figure(figcont) 
    sns.distplot(x_k_1, bins = 50, rug = False, kde = False,
                 hist = True, norm_hist = True ,
                 hist_kws = {'histtype':'stepfilled', 'linewidth':1,
                             'alpha':1, 'color':'silver'}) 

    plt.plot(x0+np.mean(x_k),last_figure_y,
             color = 'darkred', linewidth = '3',linestyle = '-')
    plt.plot(x0+np.mean(x_k),prediction_density(x0,gamma),
             color = 'darkblue', linewidth = '3',linestyle = '--')
    
    plt.xlim(-3.5,8)
    plt.ylim(0,0.8)
    plt.title(r"$k=0$",fontsize = 20)
    plt.legend( [r"$\pi(x_{k+1}\mid D_{k+1})$",r"$\pi(y_{k+1}\mid \bar{x}_{k+1})$"],
                fontsize = 15)
    plt.tick_params(labelsize=15) 
    plt.savefig("chap4_ex3_4.eps")
    