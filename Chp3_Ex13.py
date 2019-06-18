# -*- coding: utf-8 -*-
"""
Chapter 3:
            Example 13: Gibbs sampling
                problem: blind deconvolution problem

@author: XiningXu
"""

import numpy as np
import matplotlib.pyplot as plt

def phi_con(t,sigma):
    u = - t**2 / 2
    u = u / (sigma**2)
    d = 2*np.pi*(sigma**2)
    return np.exp(u) / (np.sqrt(d))

def kernel(t,K=5):
    sigma = 0.02 * (1 + np.arange(K))
    phi = [phi_con(t, sigma[0])]
    for k in range(1,K):  
        phi.append( phi_con(t, sigma[k])-phi_con(t, sigma[k-1]) )
    return phi

def test_fun(t):
    f = np.zeros(t.shape)
    out1  = np.logical_and(t>=0.3,t<=0.6)
    f[out1] = np.ones(t[out1].shape)
    return np.array(f)


if __name__=='__main__':
    figcont = 0
    alpha = 0.08
    beta = 100
    N = 100
    K = 5
    sigma = 0.02 * (1 + np.arange(K))
    v0 = np.array([1,1,1,1,1]).reshape((K,1))
    
    t0 = np.array(np.linspace(0,1,N+1)).reshape((N+1,1))
    x0 = test_fun(t0)
    dt = 1/N
    
    DT = np.array(t0-(t0.T))
    
    w = [dt/2]
    [w.append(dt) for i in range(N-1)]
    w.append(dt/2)
    w= np.array(w).T
    
    Ak = []
    Ak.append(w*np.array(phi_con(DT, sigma[0])))
    for i in range(K-1):
        A = phi_con(DT,sigma[i+1])- phi_con(DT, sigma[i])
        Ak.append(np.array(w*A))
    
    A = w*phi_con(DT,0.025)
    
    y = np.dot(A, x0)              
    gamma = 0.01*max(np.abs(y))   
    np.random.seed(0)
    e = np.random.normal(0,gamma,(N+1,1))
    y = y + e 
    
    """
    initial point
    """
    X = 0
    for k in range(K):
        X = X + v0[k]*Ak[k]
    
    X = np.vstack((X/gamma,np.eye(N+1)/alpha))
    Y = np.vstack((y/gamma,np.zeros((N+1,1))))
    x = np.dot(np.linalg.pinv(X),Y)
    
    x0 = np.array(x)
    """
    CM Estimate
    """
    cont_max = 5000
    samples_x = np.zeros((cont_max,x0.size))
    samples_v = np.zeros((cont_max,v0.size))
    
    L = np.eye(K)-np.eye(K,k=-1)
    YV = np.vstack((y/gamma, beta*np.dot(L,v0)))
    YX = np.vstack((y/gamma, np.zeros((N+1,1))))
    S = np.zeros((K,N+1))     
        
    for i in range(cont_max):
        Yv = YV + np.random.normal(0,1,(N+K+1,1))
        """
        update vector v
        """
        for k in range(K):
            S[k] = np.dot(Ak[k],x).T
        Sv = np.vstack((S.T/gamma, beta*L))
        v = np.dot(np.linalg.pinv(Sv),Yv)
        samples_v[i] = np.array(v.T)
        """
        updata vector x
        """
        Yx = YX + np.random.normal(0,1,(2*(N+1),1))
        Av = 0
        for k in range(K):
            Av = v[k]*Ak[k]
        Av = np.vstack((A/gamma,np.eye(N+1)/alpha))
        x = np.dot(np.linalg.pinv(Av),Yx)
        
        samples_x[i] = np.array(x.T)
        
    cm = np.mean(samples_x,axis = 0)  
    v_cm = np.mean(samples_v,axis = 0)
    
    """
    image of CM estimates
    """
    figcont += 1
    plt.figure(figcont)
    plt.plot(t0,test_fun(t0),'k', linewidth = '3' )
    plt.plot(t0,x0, 'b',linewidth = '3')
    plt.plot(t0,cm,'r', linewidth = '4')
    plt.xlim(0,1)
    plt.ylim(-0.55,2)
    plt.tick_params(labelsize=15)
    plt.title("CM estimate", fontsize = 15)
    plt.legend(["True","Initial","CM estimate"], fontsize=12)
    #plt.savefig("chap3_ex13_" + str(figcont) + ".eps")
    """
    CM estimate of kernel
    """
    s = np.linspace(-0.5,0.5,200)
    phi_cm = v_cm[0]*phi_con(s,sigma[0])
    phi_in = v0[0]*phi_con(s,sigma[0])
    for k in range(K-1):
        phi_cm = phi_cm + v_cm[k+1]*(phi_con(s,sigma[k+1])-phi_con(s,sigma[k]))
        phi_in = phi_in + v0[k+1]*(phi_con(s,sigma[k+1])-phi_con(s,sigma[k]))
        
    figcont += figcont
    plt.figure(figcont)
    plt.plot(s, phi_con(s,0.025),color = 'k',linewidth='3')
    plt.plot(s, phi_in,'b',linewidth = '4')
    plt.plot(s, phi_cm,'r', linewidth = '4')
    plt.xlim(-0.5,0.5)
    plt.ylim(-1,17)
    plt.tick_params(labelsize=15)
    plt.title("CM estimate for kernel", fontsize = 15)
    plt.legend(["True kernel","Initial","CM estimate"], fontsize=12)
    #plt.savefig("chap3_ex13_" + str(figcont) + ".eps")
    
    
    """
    MAP Estimate
    """
    cont = 0
    x = np.array(x0)
    while cont < 20:
        """
        update vector v
        """
        for k in range(K):
            S[k] = np.dot(Ak[k],x).T
        Sv = np.vstack((S.T/gamma, beta*L))
        v = np.dot(np.linalg.pinv(Sv),YV)
        """
        updata vector x
        """
        Av = 0
        for k in range(K):
            Av = v[k]*Ak[k]
        Av = np.vstack((A/gamma,np.eye(N+1)/alpha))
        x = np.dot(np.linalg.pinv(Av),YX)
        cont += 1

        
    figcont += 1
    plt.figure(figcont)
    plt.plot(t0,test_fun(t0),'k', linewidth = '3' )
    plt.plot(t0,x0, 'b-.',linewidth = '3')
    plt.plot(t0,x,'r', linewidth = '4')
    plt.xlim(0,1)
    plt.ylim(-0.55,2)
    plt.tick_params(labelsize=15)
    plt.title("MAP estimate", fontsize = 15)
    plt.legend(["True","Initial","MAP estimate"], fontsize=12)
    #plt.savefig("chap3_ex13_" + str(figcont) + ".eps")
        
    s = np.linspace(-0.5,0.5,200)
    phi_map = v[0]*phi_con(s,sigma[0])
    for k in range(K-1):
        phi_map = phi_map + v[k+1]*(phi_con(s,sigma[k+1])-phi_con(s,sigma[k]))
            
    figcont += 1
    plt.figure(figcont)
    plt.plot(s, phi_con(s,0.025),color = 'k',linewidth='3')
    plt.plot(s, phi_in,'b',linewidth = '4')
    plt.plot(s, phi_map,'r', linewidth = '4')
    plt.xlim(-0.5,0.5)
    plt.ylim(-1,17)
    plt.tick_params(labelsize=15)
    plt.title("MAP estimate for kernel", fontsize = 15)
    plt.legend(["True kernel","Initial","MAP estimate"], fontsize=12)
    #plt.savefig("chap3_ex13_" + str(figcont) + ".eps")
           

   