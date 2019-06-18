# -*- coding: utf-8 -*-
"""
Chapter 3:
            Example 3.4: a two-dimensional image deblurring problem

@author: Xining Xu 17110180016
"""

import numpy as np
import matplotlib.pyplot as plt

"""
    produce random parameters
"""
def prod_a(l,r):
    a = (r-l)*np.random.rand() + l;
    return a

def prod_r(a,b):
    r = (b-a)*np.random.rand() + a;
    return r

def prod_p0(mu,sigma):
    p0 = np.random.normal(mu, sigma, 2);
    return p0;



def heaviside(x):
    if x>=0:
        return 1
    else:
        return 0 
    
def distance(a):
    return np.linalg.norm(a,ord=2) 






if __name__=='__main__':  
    
    
    N = 500
    k = 40
    figcont = 0
    pk = 0.5 + np.arange(k) 
    pk /=k
    
    
    """
        build image matrix
    """
    X = np.zeros((k*k,N)) 
    for t in range(N):
        a = prod_a(0.8,1.2)
        r = prod_r(0.2,0.25)
        p0 = prod_p0(0.5,0.02)
        
        P = np.zeros((k,k))
        for i in range(k):
            for j in range(k):
                x,y = pk[i],pk[j]
                d = distance([p0[0]-x, p0[1]-y])
                P[i][j] = a*heaviside(r-d)
        X[:,t] = P.reshape((1,k*k), order='F')
        
    
    """
        Compute singular system for covariance matrix 
    """
    X_mean = np.mean(X,axis=1)
    P_mean = X_mean.reshape((k,k),order='F')
    X_cov = np.cov(X)
    U,sigma,VT = np.linalg.svd(X_cov)
    sigma = sigma[0:400]
    
    print("-------------------------------------------------------")
    print("300 largest eigenvalues of estimated covariance matrix:")
    print("-------------------------------------------------------")
    figcont+=1
    plt.figure(figcont)
    plt_sigma = plt.semilogy(np.arange(400)+1, sigma,'.')
    plt.ylim([1e-20,1e+5])
    plt.xlim([0,300])
    plt.title("300 largest eigenvalues of covariance matrix", fontsize=12)
    #plt.savefig("./chap3_ex4_"+str(figcont)+".eps")
    plt.show()
    
    
    print("------------")
    print("Mean(image):")
    print("------------")
    figcont+=1
    plt.figure(figcont)
    plt.imshow(P_mean,cmap=plt.cm.gray, interpolation='nearest')
    #plt.savefig("./chap3_ex4_"+str(figcont)+".eps")
    plt.show()
    
    
    """
        Eigenvectors correponding to the 5 largest eigenvalues
    """
    print("-------------------------------------------------------")
    print("Eigenvectors correponding to the 5 largest eigenvalues:")
    print("-------------------------------------------------------")
    for t in range(5):
        figcont+=1
        plt.figure(figcont)
        sv = U[:,t]
        sv = sv.reshape((k,k), order = 'F')
        plt.imshow(sv, cmap=plt.cm.gray, interpolation='nearest')  
        #plt.savefig('./chap3_ex4_'+ str(figcont)+'.eps')
    
    
    
    
