# -*- coding: utf-8 -*-
"""
Supplementary Example:  
        
    Kaczmarz Iteration for one-dimensional deconvolution problem 

@author: XiningXu 17110180016
"""
import numpy as np
import matplotlib.pyplot as plt

def conv_ker(a,t):
    return np.exp(-a*np.abs(t))

def test_fun(t):
    return t*(1-t)

def conv_g(a,s):
    gs1 = 2*s*(1-s)/a 
    gs2 = ( np.exp(-a*s) + np.exp(-a*(1-s)) ) / (a**2) 
    gs3 = 2*( np.exp(-a*s) + np.exp(-a*(1-s)) - 2)/ (a**3)
    return gs1+gs2+gs3

def iter_Kaczmarz(A, x, y):
    R,_ = A.shape
    for j in range(R):
        aj = np.reshape(A[j],(1,-1))
        x_t = np.dot(aj.T, y[j]-np.dot(aj,x))
        x = x + x_t/(np.dot(aj,aj.T))
    return x
     

    
if __name__=='__main__':
        
    a = 20
    H = 100
    W = 80
    figcont = 0
    """
        build system 
    """
    s = (1/H) * (np.arange(100+1))
    t = (1/W) * (np.arange(80+1))
     
    gs = conv_g(a,s.reshape((-1,1)))
    ft = test_fun(t.reshape((-1,1)))

    A = -s.reshape((s.size,1)) + t.reshape((1,t.size)) 
    A = conv_ker(a,A)/W

    
    std = 0.05*max(gs)
    np.random.seed(0)
    rand_e = np.random.normal(0, std, (H+1,1))
    
    y = gs + rand_e
    
    f_LS = np.dot(np.linalg.pinv(A),y)  

    
    epsilon = np.sqrt(W+1)*std
    
    print("----------------------------")
    print("Kaczmarz Iteration solution:")
    print("----------------------------")
    figcont += 1
    plt.figure(figsize=(8,4.8))
    true_func,=plt.plot(t,ft,'r-.',linewidth=4)
    min_nr,=plt.plot(t,f_LS,'gray',linewidth=1.5)
    
    
    f = np.zeros((W+1,1))
    error = np.linalg.norm(y-np.dot(A,f))
    cont = 0
    while error > epsilon:
        f = iter_Kaczmarz(A, f, y)
        cont += 1
        plt.plot(t,f,'k',linewidth= 2)
        error = np.linalg.norm(y-np.dot(A,f))
    
        
    final_iter,=plt.plot(t,f,'deepskyblue',linestyle='--',linewidth=4)
    plt.xlim(0,1)
    plt.ylim(-1.5,2.2)
    plt.ylabel("f(t)", fontsize=18)
    plt.xlabel("t", fontsize=18)
    plt.title("Number of iterations = " + str(cont), fontsize=18)
    plt.tick_params(labelsize=15)
    plt.legend([true_func, min_nr,final_iter],
               ["true","minimum norm","final iterated solution"],
               loc='upper right', fontsize=10)
    #plt.savefig('./MyExample1_1.eps')
    plt.show()
      
    print("")
    print("The Landweber–Fridman iteration progresses slower than several other iterative methods."
          +"The slow convergence of the method is sometimes argued to be a positive feature of the algorithm,"
          +"since a fast progress may bring us quickly close to the minimum norm solution that is usually nonsense.")