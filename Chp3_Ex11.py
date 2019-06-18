# -*- coding: utf-8 -*-
"""
Chapter 3: 
            Example 3.11: Metropolis–Hastings algorithm

@author: XiningXu 17110180016
"""

import numpy as np
import matplotlib.pyplot as plt


def fun_pi(x):
    """
        density function 
    """
    x1 = x[0]
    x2 = x[1]
    pi_x = -10*(x1**2-x2)**2 - (x2-1/4)**4
    return np.exp(pi_x)


def q_x(x,y,gamma):
    """
        proposal distribution
    """
    q = np.linalg.norm(x-y)
    q = q**2 / (gamma**2)
    return np.exp(-q/2)


def Metro_Hast(x0, gamma, K):
    """
        Metropolis–Hastings
    """
    X = np.zeros((K,2))
    x = x0
    X[0] = x.T
    
    for k in range(1,K):
        pi_x = fun_pi(x)
        w = np.random.normal(0,gamma,(2,1))
        y = x+w
        pi_y =fun_pi(y)
        alpha = np.min((pi_y/pi_x, 1))
        u = np.random.uniform(0,1)
        if u < alpha:
            x = y
            X[k] = x.T
        else:
            X[k] = x.T
    
    return X
        
if __name__=='__main__':        
        
    N = 100
    h = 1/N
    figcont = 0
    
    x = np.linspace(-2+h/2, 2-h/2, N)
    y = np.linspace(-1+h/2, 2-h/2, N)
    X, Y = np.meshgrid(x, y)
    
    
    P = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            P[i][j] = fun_pi((x[i],y[j]))
     
        
    """
        contour
    """
    figcont += 1
    fig1 = plt.figure(figcont)
    fig1.figsize = (6,4.5)
    plt.contourf(X, Y, P.T, 10, cmap = plt.cm.Blues_r)
    plt.contour(X, Y, P.T, 10, colors = 'r', linewidth = 0.1)
    plt.xlim(-2,2)
    plt.ylim(-1,2)
    plt.tick_params(labelsize='15')
    plt.title("contour plot of the density",fontsize='18')
    # plt.savefig("./chap3_ex11_"+str(figcont)+".eps")
    plt.show()
    
    
    
    """
        Metropolis-Hastings
    """
    x0 =np.array([[1.5],[-0.8]])
    cont_max = 5000
    
    gamma_list =[0.01,0.05,0.1,1]
    for gamma in gamma_list:
        x_new = Metro_Hast(x0, gamma, cont_max)
        x_new = x_new.T
        
        print("-------------------------------------------------------------")        
        print("Random walk Metropolis–Hastings runs with gamma="+ str(gamma))
        print("-------------------------------------------------------------")
        figcont += 1
        fig2 = plt.figure(figcont)
        fig2.figsize = (8,6)
        plt.contourf(X, Y, P.T, 10, alpha =0.8, cmap = plt.cm.Blues_r)
        plt.plot(x_new[0],x_new[1],'r.',markersize ='2')
        plt.xlim(-2,2)
        plt.ylim(-1,2)
        plt.tick_params(labelsize='15')
        plt.title("$\gamma = $"+ str(gamma),fontsize='18')
        #plt.savefig("./chap3_ex11_" + str(figcont)+".eps")
        plt.show()

        print("-------------------------------------------------------------")        
        print("Convergence diagnostics for the component x1 with gamma="+ str(gamma))
        print("-------------------------------------------------------------")
        figcont += 1
        fig3 = plt.figure(figcont)
        fig3.figsize = (6,4.5)
        plt.plot(np.arange(cont_max),x_new[0],'k',linewidth='1.5')
        plt.xlim(0,cont_max)
        plt.ylim(np.min(x_new[0])*1.1,2)
        plt.tick_params(labelsize='15')
        plt.title("$\gamma = $"+ str(gamma),fontsize='18')
        #plt.savefig("./chap3_ex11_" + str(figcont)+".eps")
        plt.show()
        
    print("--------------------------------------")
    print("remark1: When the step length is small, the process explores the density slowly,"
          +"new proposals are accepted frequently, "
          +"and the sampler moves slowly but steadily.")
    print("remark2: It takes a while before the chain starts to properly sample the distribution.")
    print("When the step length is small, the chain has a long tail "
          +"from the starting value to the proper support of the density.")
