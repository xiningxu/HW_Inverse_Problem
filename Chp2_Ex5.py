# -*- coding: utf-8 -*-
"""
Chapter2 
            Example 2.5 Tikhonov regularization Method
                    problem: image deblurring problem (deconvolution problem)
                    
@author: XiningXu 17110180016
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def conv_phi(a,x):
    """
        convolution kernel
    """
    return np.exp(-a*np.linalg.norm(x))

def build_matrix(N,Delta):
    p = 0.5*Delta + Delta*np.arange(N)
    A = np.zeros((N*N,N*N))
    
    for R in range(N*N):
        Ar = np.zeros((N,N))
        j = R//N
        k = R%N
        pjk = np.array([p[j],p[k]])
        for l in range(N):
            for m in range(N):
                plm = np.array([p[l],p[m]])
                Ar[l][m]= conv_phi(a,pjk-plm)
        A[R] = np.reshape(Ar,(1,N*N))
    return A*Delta**2

def Tikhonov(A,y,N,delta):
    """
        Tikhonov Regularization Method
    """
    delta_mat = delta*np.eye(N*N)
    K = np.vstack((A,delta_mat))
    Y = np.vstack((y,np.zeros((N*N,1))))
    x_delta = np.dot(np.linalg.pinv(K),Y)
    r  = np.linalg.norm(np.dot(A,x_delta)-y)
    return x_delta, r





if __name__=='__main__':
      
    
    a = 20
    N = 32
    Delta = 1/N
    figcont = 0
    
    
    
    """ 
        read image
    """
    image_path = "./chap2_ex5_1.jpg"
    I = Image.open(image_path)
    L = I.convert('L')
    img = np.array(L)
    
    print("--------------------------")
    print("Oringinal image:")
    print("--------------------------")
    figcont += 1
    plt.figure(figcont)
    plt.imshow(img,cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    # plt.savefig("./chap2_ex5_" + str(figcont) + ".eps")
    plt.show()
    
    
    """ 
        build noisy blurred image
    """
    x = np.reshape(img,(N*N,1),order ='F')
    A =build_matrix(N,Delta)
    g = np.dot(A,x)
    
    std = 0.01*max(g)
    np.random.seed(0)
    rand_e = np.random.normal(0, std, (N*N,1))
    y = g + rand_e 
    epsilon = N*std
    
    print("------------------------------------")
    print("Noisy blurred image(sd = 1%max(Ax)):")
    print("------------------------------------")
    figcont += 1
    plt.figure(figcont)
    plt.imshow(np.reshape(y,(N,N),order='F'),cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    # plt.savefig("./chap2_ex5_" + str(figcont) + ".eps")
    plt.show()
       
    
    """
        direct solution i.e. delta = 0
    """
    delta = 0
    x_delta,_ = Tikhonov(A,y,N,delta)
        
    print("--------------------------------")
    print("Direct solution(i.e. delta = 0):")
    print("--------------------------------")
    figcont += 1
    plt.figure(figcont)
    plt.imshow(np.reshape(x_delta,(N,N),order='F'),cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    # plt.savefig("./chap2_ex5_" + str(figcont) + ".eps")
    plt.show()
    
    
    """
        Discrepancy v.s. regularization parameter
    """
    delta_list = np.linspace(8*1e-5,5*1e-3,50)
    r_norm = []
    for delta in delta_list:
        x_delta, r = Tikhonov(A,y,N, delta)
        r_norm.append(r)    
    
    print("------------------------------------------")
    print("Discrepancy v.s. regularization parameter:")
    print("------------------------------------------")
    plt.figure(figsize=(8,4))
    plt.plot(np.array(delta_list),epsilon*np.ones(delta_list.shape),
             'k:',linewidth=4)
    plt.plot(np.array(delta_list), r_norm, linewidth = 4 )
    plt.show()
    
    
    """
        bisection method to meet Morozov discrepancy principle
    """
    l = 8*1e-4; r = 1e-3;
    tol = 1e-4;
    d = 1
    cont = 0
    while d>tol:
        _, dl = Tikhonov(A, y, N, l)
        _, dr = Tikhonov(A, y, N, r)
        _, d_mid = Tikhonov(A, y, N, 0.5*(l+r))
        flag = (dl-epsilon)*(d_mid-epsilon)
        if flag<0:
            r = 0.5*(l+r)
        else:
            l = 0.5*(l+r)
        cont += 1
        d = np.abs(dl-dr)
    delta_opt = (l+r)/2   
    x_opt,_ = Tikhonov(A, y, N, delta_opt)
    
    
    delta_list_mark = [0.0001,0.0004,delta_opt,0.002, 0.004]
    r_norm_mark =[]
        
    for delta in delta_list_mark:
        x_delta, r = Tikhonov(A,y,N, delta)
        r_norm_mark.append(r) 
        
        figcont += 1
        plt.figure(figcont)
        plt.imshow(np.reshape(x_delta,(N,N),order='F'),cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("$\delta=$"+ str(delta))
        #plt.savefig("./chap2_ex5_"+ str(figcont)+".eps",fontsize =20)
        plt.show()
        
    print("--------------------------------------------------------")
    print("When the regularization parameter is small, outcome is noisy(underregularized)")
    print("For large values, the results get again blurred, that is overregularized solution.")


    print("------------------------------------------")
    print("Discrepancy v.s. regularization parameter:")
    print("------------------------------------------")         
    plt.figure(figsize=(8,4))
    plt.plot(np.array(delta_list),epsilon*np.ones(delta_list.shape),
             'k:',linewidth=4)
    plt.plot(np.array(delta_list), r_norm, linewidth = 4 )
    plt.plot(np.array(delta_list_mark), r_norm_mark, 'r*',markersize=12)
    plt.tick_params(labelsize=12)
    plt.xlabel("$\delta$",fontsize=15)
    plt.ylabel('$\|Ax_{\delta}-y\|$',fontsize=12)
    plt.xlim((0,np.max(delta_list)))
    # plt.savefig("./chap2_ex5_"+str(cont)+".eps")
    plt.show()
        
    
