# -*- coding: utf-8 -*-
"""
Chapter 2:
            Example 2.6: Landweber-Fridman Iteration
                problem: one-dimensional deconvolution problem
                
@author: XiningXu 17110180016
"""
import numpy as np
import matplotlib.pyplot as plt

def conv_ker(a,t):
    """
        convolution kernel
    """
    return np.exp(-a*np.abs(t))


def test_fun(t):
    """
        test function
    """
    return t*(1-t)


def conv_g(a,s):
    """
        explicit expression for the convolution of test function 
    """
    gs1 = 2*s*(1-s)/a 
    gs2 = ( np.exp(-a*s) + np.exp(-a*(1-s)) ) / (a**2) 
    gs3 = 2*( np.exp(-a*s) + np.exp(-a*(1-s)) - 2)/ (a**3)
    return gs1+gs2+gs3


def iter_LanFri(A, x, y, beta):
    """
        Landweber-Fridman Iteration
    """
    y_new = beta*np.dot(A.T,y)
    x_new = x - beta*np.dot(A.T,np.dot(A,x))
    return x_new + y_new


def iter_Kaczmarz(A, x, y):
    """
        Kaczmarz Iteration
    """
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
    
    
    """
        observed data
    """
    print("--------------------------------")
    print("Noisy and noiseless observation:")
    print("--------------------------------")
    figcont += 1
    plt.figure(figcont)
    plt.figure(figsize=(8,4.8))
    l1_noiseless,=plt.plot(s,gs,'r',linewidth=4)
    l2_noisy,=plt.plot(s,y,'k',linewidth=3)
    plt.xlim(0,1)
    plt.ylim(0,0.03)
    plt.ylabel("g(s)", fontsize=18)
    plt.xlabel("s", fontsize=18)
    plt.title("Data",fontsize = 18)
    plt.legend([l1_noiseless,l2_noisy],["noiseless","noisy"],loc='upper right',
               fontsize=13)
    plt.tick_params(labelsize=15)
    #plt.savefig("./chap2_ex6_"+ str(figcont)+".eps")
    plt.show()
    
    
    """
        direct calculation of the minimum norm solution f† = A†g 
    """
    f_LS = np.dot(np.linalg.pinv(A),y)
    print("-------------------------------------------------")
    print("Original test function and minimum norm solution:")
    print("-------------------------------------------------")    
    figcont +=1
    plt.figure(figcont)
    plt.figure(figsize=(8,4.8))
    l1_true,=plt.plot(t,ft,'r', linewidth=4)
    l2_mns,=plt.plot(t,f_LS,'k',linewidth=2.5)
    plt.xlim(0,1)
    plt.ylim(-1.5,2.5)
    plt.ylabel("f(t)", fontsize=18)
    plt.xlabel("t", fontsize=18)
    plt.title("Minimum norm solution",fontsize = 18)
    plt.legend([l1_true,l2_mns],["true","minimum norm"],loc='upper right',
               fontsize=13)
    plt.tick_params(labelsize=15)
    #plt.savefig("./chap2_ex6_"+ str(figcont)+".eps")
    plt.show()
    
    print("Minimum norm solution is essentially pure noise, "
          + "showing that some form of regularization is required")
    
    
    """
        solutions through Landweber-Fridman Iteration Method
    """
    beta_max = 2/(np.linalg.norm(A,ord = 2)**2)
    beta = 0.1*beta_max
    epsilon = np.sqrt(W+1)*std
    
    print("-------------------------------------")
    print("Landweber-Fridman Iteration solution:")
    print("-------------------------------------")
    figcont += 1
    plt.figure(figsize=(8,4.8))
    true_func, = plt.plot(t,ft,'r',linestyle='-.', linewidth=4)
    
    f = np.zeros((W+1,1))
    error = np.linalg.norm(y-np.dot(A,f))
    cont = 0
    
    while error > epsilon:
        f = iter_LanFri(A, f, y, beta)
        cont += 1
        plt.plot(t,f,'k',linewidth= 1.5)
        error = np.linalg.norm(y-np.dot(A,f)) 
        
    
    final_iter, = plt.plot(t,f,'deepskyblue',linestyle='--',linewidth=4)
    plt.xlim(0,1)
    plt.ylim(0,0.35)
    plt.ylabel("f(t)", fontsize=18)
    plt.xlabel("t", fontsize=18)
    plt.title("Number of iterations = " + str(cont), fontsize=18)
    plt.tick_params(labelsize=15)
    plt.legend([true_func, final_iter],["true","final iterated solution"],loc='upper right',
           fontsize=15)
    #plt.savefig("./chap2_ex6_"+ str(figcont)+".eps")
    plt.show()
    print("Final solution is marked by blue line."
          +" Landweber–Fridman iteration convergences slowly.")
    
    
    """    
    ==========================================================================
            Supplement Example 1 : Kaczmarz Iteration
    ==========================================================================
    """
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