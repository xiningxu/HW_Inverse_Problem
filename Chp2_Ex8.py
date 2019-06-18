# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:59:32 2019

@author: xinin
"""

# -*- coding: utf-8 -*-
"""
Chapter2 
            Example 8  
                    problem: Laplace inversion problem
                    
@author: XiningXu 17110180016
"""
from __future__ import division
from scipy.special.orthogonal import p_roots

import numpy as np
import matplotlib.pyplot as plt
 
def gauss_nodes(n,a,b):
    """
        Get Gaussian Legraend nodes and weights
    """
    [x,w] = p_roots(n)
    x = (b-a)*x +(b+a)
    x /= 2
    w *= 0.5*(b-a)
    return x, w


def s_nodes(n):
    s = np.arange(1,n+1) 
    s = (-1 + (s-1)/20) * np.log(10)
    s = np.exp(s)
    return s


def test_fun_lap(s):
    Lf = (2- 3*np.exp(-s) + np.exp(-3*s)) / (2*(s**2))
    return Lf


def test_fun(t):
    ft =np.zeros(t.size)
    out1 = np.logical_and(0<=t, t<1)
    ft[out1] = t[out1]
    out2 = np.logical_and(1<=t, t<3)
    ft[out2] = 3/2-0.5*t[out2]
    return ft


def get_system(a,b,n):
    t,w = gauss_nodes(n,a,b)
    x = test_fun(t)
    s = s_nodes(n)
    A = w*np.exp(-np.reshape(s,(-1,1))*np.reshape(t,(1,-1)))
    y = np.matmul(A,x)
    return s, t , x, y, A, w

def CG(A,x,y):
    r = y-np.dot(A,x)
    s = r
    
    return s
    
    
if __name__=='__main__':
    
    
    N = 40
    a = 0
    b = 5
    figcont = 0
    s, t, x, y, A,w = get_system(a,b,N)
    f = test_fun(np.linspace(0,5,500))
       
    sd = 0.01*np.linalg.norm(y,ord=np.inf)
    np.random.seed(20)
    e = np.random.normal(0,sd,y.shape)
    y = y-e
    r0_norm = np.linalg.norm(y-np.dot(A,x), ord=2)
    
    B = np.matmul(A.T,A)
    y = np.dot(A.T,y)
    x = np.zeros(x.size)
    
    r = y - np.dot(B,x)
    r_norm = []
    s = r
    for i in range(9):
        alpha = np.linalg.norm(r, ord =2)**2/np.matmul(s.T,np.matmul(B,s))
        x_new = x + alpha*s
        r_new = r - alpha* np.dot(B,s)
        beta = np.dot(r_new.T, r_new)/np.dot(r.T, r)
        s_new = r_new + beta*s
        x = x_new
        r = r_new
        s = s_new
        r_norm.append(np.linalg.norm(y-np.dot(B,x),ord = 2))
        
        figcont += 1
        plt.figure(figsize = (5,5))
        plt.plot(np.linspace(0,5,500),f,linewidth='3',color = 'k')
        plt.plot(t,x,linewidth='3',color='red',linestyle = '--')
        plt.xlim(0,5)
        plt.ylim(-0.8,1.6)

        plt.title("Iterations = " + str(i+1),fontsize=20)
        plt.tick_params(labelsize=18)
        #plt.savefig("./chap2_ex8_"+ str(i+1)+".eps")
    
    figcont += 1
    r_norm = np.array(r_norm)
    plt.figure(figcont)
    plt.plot(np.arange(9)+1,r_norm,'k',linewidth ='3')
    plt.plot(np.arange(9)+1,r_norm,'r*',markersize ='10') 
    plt.plot(np.arange(9)+1, r0_norm * np.ones(r_norm.size), 
             'b--',linewidth = '3')
    plt.xlim(0,10)
    plt.ylim(-0.02,0.16)
    plt.ylabel(r'$||r||_2$',fontsize =14)
    plt.tick_params(labelsize=11)
    #plt.savefig("./chap2_ex8_"+ str(figcont)+".eps")