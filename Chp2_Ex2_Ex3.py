# -*- coding: utf-8 -*-
"""
Chapter2 
            Example 2.2 
                    problem: Laplace inversion problem
                    
            Example 2.3 TSVD Method
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


def M_TSVD(A,y,p):
    """
        TSVD step
    """
    N = np.size(y)
    U,sigma,VT = np.linalg.svd(A)
    sigma_p = 1/sigma[0:p]
    d_line = np.append(sigma_p,np.zeros((N-p,1)))
    sigma_pinv = np.diag(d_line)
    
    x_pinv = np.dot(U.T, y)
    x_pinv = np.dot(sigma_pinv, x_pinv)
    x_pinv = np.dot(VT.T, x_pinv)
    return x_pinv




    
if __name__=='__main__':
    
    
    N = 40
    a = 0
    b = 5
    figcont = 0
    s, t, x, y, A,w = get_system(a,b,N)
    
    """    
    ==========================================================================
                                Example 2.2
    ==========================================================================
    """
    
    """
        Plot test function f(t)
    """
    print("-------------")
    print("测试函数图像：")
    print("-------------")
    figcont += 1
    fig = plt.figure(figsize=(8,2))
    plt.plot(np.linspace(0,5,100),
             test_fun(np.linspace(0,5,100)), linewidth = '3')
    plt.xlim(0,5)
    plt.ylabel('$f(t)$',fontsize = '20')
    plt.xlabel('$t$',fontsize = '20')
    plt.tick_params(labelsize=15)
    #plt.savefig("./chap2_ex2_"+str(figcont)+".eps")
    plt.show()
    
    
    """ 
        Plot convolution for test function   
    """
    print("---------------------")
    print("f(t)的Laplace变换图像:")
    print("---------------------")
    figcont += 1
    fig = plt.figure(figsize=(8,2))
    plt.plot(np.linspace(0,np.max(s),100),
             test_fun_lap(np.linspace(0,np.max(s),100)),linewidth = '3')
    plt.plot(s,y,'k*')
    plt.xlim(0,9)
    plt.ylabel('$\mathcal{L}f(t)$',fontsize = '20')
    plt.xlabel('$s$',fontsize = '20')
    plt.tick_params(labelsize=15)
    #plt.savefig("./chap2_ex2_"+str(figcont)+".eps")
    plt.show()
    
    
    """
        Plot solution for solving the linear system
    """
    print("--------------------------")
    print("直接求解反(离散)Laplace变换:")
    print("--------------------------")
    figcont += 1
    plt.figure(figsize=(8.2,7.5/3))
    f_lin = np.linalg.solve(A,test_fun_lap(s))
    plt.plot(t,f_lin,linewidth = '3')
    plt.xlim(0,5)
    plt.ylim(-750,750)
    plt.title('direct solution',fontsize = '20')
    plt.xlabel('$t$',fontsize = '20')
    plt.tick_params(labelsize=15)
    #plt.savefig("./chap2_ex2_"+str(figcont)+".eps")
    plt.show()
    print("直接求解线性方程组可能带来灾难性的后果.")
    
    
    
    
    """
    ==========================================================================
                                Example 2.3 TSVD 
    ==========================================================================
    """
    figcont = 0
    _, sigma,_ = np.linalg.svd(A) # singular value 
    
    print("---------------------------------------")
    print("矩阵A的奇异值图像:")
    print("---------------------------------------")
    std = 0.01*np.linalg.norm(y,ord=np.inf)
    epsilon = 1e-16
    figcont += 1
    plt.figure(figcont)
    plt.plot(np.arange(N)+1, epsilon*np.ones((N,1)),
             '--',color='k',linewidth =3)
    plt.plot(np.arange(N)+1, std*np.ones((N,1)),
             ':',color='b',linewidth =3)
    plt.plot(np.arange(N)+1, sigma,
             '*', color='r',markersize=8)
    plt.ylim(5e-18,20)
    plt.xlim(0,40)
    plt.yscale('log')
    plt.title('singular value of matrix A',fontsize = 15)   
    plt.tick_params(labelsize=15)
    #plt.savefig("./chap2_ex3_"+ str(figcont)+".eps")
    plt.show()
    print("共有23个奇异值在机器精度之上, 但第22个奇异值已经十分接近机器精度.")


    """
        reconstruction of no artificial noise obsevations 
    """
    print("-----------------------------")
    print("(不含噪声)数据TSVD方法反演结果:")
    print("-----------------------------")
    p_list = [20,21,22]
    l_set =['-','--',':']
    c_set =['b','r','k']
    lineset =[]
    
    figcont += 1
    plt.figure(figsize=(7.5,7.5/3))
    for ii in range(len(p_list)):
        lineset.extend(plt.plot(t, M_TSVD(A,y,p_list[ii]),
                             linewidth='3',linestyle=l_set[ii],color=c_set[ii]))
    plt.legend(lineset,[("p="+str(p)) for p in p_list],loc='upper right',
           fontsize=15)
    plt.xlim(0,5)  
    plt.ylim(-0.2,1.5)   
    plt.title("Reconstruction of noiseless obsevation with TSVD(p)",
              fontsize =15)
    plt.tick_params(labelsize=15)    
    #plt.savefig("./chap2_ex3_"+ str(figcont)+".eps")
    plt.show()
    print("由于第22个奇异值已经十分接近机器精度, 其反演结果明显较差.")
    
    
    """
        reconstruction of obsevations with artificial noise
    """
    print("-----------------------------")
    print("(含噪声)数据TSVD方法反演结果:"  )
    print("-----------------------------")
    p_list = [4,5,6]
    l_set =['-','--',':']
    c_set =['b','r','k']
    lineset =[]
    
    np.random.seed(20)
    e = np.random.normal(0,std,y.shape)
    figcont += 1
    plt.figure(figsize=(7.5,8/3))
    for ii in range(len(p_list)):
        lineset.extend(plt.plot(t, M_TSVD(A, y+e, p_list[ii]),
                             linewidth='3',linestyle=l_set[ii],color=c_set[ii]))
    plt.legend(lineset,[("p="+str(p)) for p in p_list],loc='upper right',
           fontsize=12)
    plt.xlim(0,5)  
    plt.ylim(-1,1.5)   
    plt.title("Reconstruction of noisy obsevation with TSVD(p)",
              fontsize =15)
    plt.tick_params(labelsize=12)
    #plt.savefig("./chap2_ex3_"+ str(figcont)+".eps")
    plt.show()
  
    
