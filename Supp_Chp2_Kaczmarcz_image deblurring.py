# -*- coding: utf-8 -*-
"""
Supplementary Example: 
    
    Kaczmarz Method for image deblurring problem 
                    
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
    
    
    """
        direct solution i.e. delta = 0
    """
    print("最小范数解")
    x_mn = np.linalg.solve(np.dot(A.T,A),np.dot(A.T,y))
    figcont += 1   
    plt.figure(figcont)
    plt.imshow(np.reshape(x_mn,(N,N),order='F'),cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    #plt.savefig("./MyExample2_" + str(figcont) + ".eps")
    plt.show()
      

    
    
    epsilon = N*std
    cont = 0
    r_norm = 2*epsilon
    
    print("Kaczmarz Iteration")  
    x_ka = np.zeros(x.shape)
    while r_norm > epsilon:
        x_ka = iter_Kaczmarz(A, x_ka, y)
        r_norm = np.linalg.norm(y- np.dot(A,x_ka),ord=2)
        cont += 1
        
    figcont += 1   
    plt.figure(figcont)
    plt.imshow(np.reshape(x_ka,(N,N),order='F'),cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title("Kaczmarz  (iterations=" + str(cont)+")",fontsize=12)
    #plt.savefig("./MyExample2_" + str(figcont) + ".eps")
    plt.show()
    

    
    
    