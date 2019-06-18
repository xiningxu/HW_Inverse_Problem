# -*- coding: utf-8 -*-
"""
Chapter 4: 
            Example 2: Kalman Filters
                problem: Linear Gaussian Problems

@author: XiningXu 17110180016
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def wave_phi(s):
    phi1 = a1*np.exp(-(s-c1)**2 / s1**2)
    phi2 = a2*np.exp(-(s-c2)**2 / s2**2)
    return phi1+phi2

def build_F_Cov(v, sn, gamma, dt):
    L = np.eye(sn+1) - np.eye(sn+1, k=-1)
    L = -v*sn*L
    F = np.eye(sn+1) - dt*L
    F = np.linalg.inv(F)
    
    E = np.zeros((sn+1,1))
    E[0] = 1
    Cov_w = alpha**2 * np.dot(E,E.T) + gamma**2*np.eye(sn+1)
    Cov_w = np.dot(F, np.dot(Cov_w,F.T))
    return F, Cov_w

def Kalmen_Filer(Y, G, F, tn, Cov_w, Cov_k, Mean_k):
    X = []
    
    for i in range(tn):
        Mean_k1 = np.dot(F, Mean_k.reshape((sn+1),1))
        Cov_k1 = np.dot(F, np.dot(Cov_k, F.T)) + Cov_w
        
        K = np.dot(G,np.dot(Cov_k1,G.T)) + Cov_v
        K = np.linalg.inv(K)
        K = np.dot(Cov_k1,np.dot(G.T, K))
        
        y = Y[i+1].reshape(sn+1,1)
        Mean_k = y - np.dot(G,  Mean_k1)
        Mean_k = Mean_k1 + np.dot(K, Mean_k)
        Cov_k = np.eye(sn+1)-np.dot(K, G)
        Cov_k = np.dot(Cov_k, Cov_k1)
        
        X.append(np.array(Mean_k))
        
    X = np.array(X)
    X = np.squeeze(X)
    return X
    

if __name__=='__main__': 
    figcont = 0
    a1 = 1.5; a2 = 1.0;
    s1 = 0.08; s2 = 0.04;
    c1 = 0.1; c2 = 0.25;
    c = 0.04
    
    gamma1 = 0.1
    gamma2 = 1e-4
    alpha = 0.1
    
    sn = 200
    tn = 20
    s = np.linspace(0,1,sn+1).reshape((sn+1,1))
    t = np.linspace(0,20,tn+1).reshape((1,tn+1))
    Space, Time = np.meshgrid(s,t)
   
    s_i = s-c*(t)
    wave_surf =  wave_phi(s_i)
    wave_surf = wave_surf.T



    """
        plot original wave
    """
    figcont +=1
    fig = plt.figure(figsize=(8,4)) 
    ax = Axes3D(fig)
    ax.plot_surface(Space, Time, wave_surf, 
                    facecolors=cm.Blues_r(wave_surf/4*wave_surf.max()),
                           antialiased=False,shade =False)
    ax.view_init(elev = 40,azim= 235)
    ax.set_zlim(-2,2)
    ax.set_zticks(np.array([0,1,2]))
    ax.grid(False)
    plt.xlim(-0.05,1)
    plt.ylim(-0.5,20)
    plt.xlabel("space(s)",size = 12)
    plt.ylabel("time(t)",size = 12)   
    plt.xticks(np.linspace(0,1,6))
    plt.yticks(np.linspace(0,tn,tn//5+1))
    plt.tick_params(labelsize='10')
    plt.title("true wave", fontsize = 15)
    plt.grid()
    #plt.savefig("chap4_ex2_"+str(figcont)+".eps")
    plt.show()
    
    
    
    
    """
        Add Noise and plot the observation
    """
    Mean_v = np.zeros(sn+1)
    st = 0.01*np.max(wave_surf)
    Cov_v =  st*np.eye(sn+1)
    e = np.random.multivariate_normal(Mean_v, Cov_v, tn+1)
    Y = np.array(wave_surf + e)
    G = np.eye(sn+1)
    """
        plot observation of the wave
    """
    figcont +=1
    fig = plt.figure(figsize=(8,4)) 
    ax = Axes3D(fig)
    ax.plot_surface(Space, Time, Y, 
                    facecolors=cm.Blues_r(wave_surf/4*wave_surf.max()),
                           antialiased=False,shade =False)
    ax.view_init(elev = 40,azim= 235)
    ax.set_zlim(-2,2)
    ax.set_zticks(np.array([0,1,2]))
    ax.grid(False)
    plt.xlim(-0.05,1)
    plt.ylim(-0.5,20)
    plt.xlabel("space(s)",size = 12)
    plt.ylabel("time(t)",size = 12)   
    plt.xticks(np.linspace(0,1,6))
    plt.yticks(np.linspace(0,tn,tn//5+1))
    plt.tick_params(labelsize='10')
    plt.title("Observation", fontsize = 15)
    plt.grid()
    #plt.savefig("chap4_ex2_"+str(figcont)+".eps")
    plt.show()
    
    
    
    
    """
        Kalman Filter: Random Walk Model
    """
    F = np.eye(sn+1)
    Cov_w = gamma1*np.eye(sn+1)
    
    
    Cov_k = 1*np.eye(sn+1)
    Mean_k = np.zeros(sn+1)

    X = Kalmen_Filer(Y, G, F, tn, Cov_w, Cov_k, Mean_k)
    
    t = np.linspace(1,20,tn).reshape((1,tn))
    Space, Time = np.meshgrid(s,t)
    
    figcont += 1
    fig = plt.figure(figcont, figsize=(8,4)) 
    ax = Axes3D(fig)
    
    surf = ax.plot_surface(Space, Time, X, facecolors=cm.Blues_r(X/4*X.max()),
                           antialiased=False,shade =False)
    
    ax.view_init(elev = 45, azim= 235)
    ax.set_zlim(-2,2)
    ax.set_zticks(np.array([-2,0,2]))
    ax.grid(False)
    plt.xlim(-0.05,1)
    plt.ylim(-0.5,20)
    plt.xlabel("space(s)",size = 12)
    plt.ylabel("time(t)",size = 12)   
    plt.xticks(np.linspace(0,1,6))
    plt.yticks(np.linspace(0,tn,tn//5+1))
    plt.tick_params(labelsize='10')
    plt.title("random walk model", fontsize = 15)
    plt.grid()
    #plt.savefig("chap4_ex2_"+str(figcont)+".eps")
    plt.show()
    
    
    
    
    
    """
        Kalman Filter: wave propagation model
                
            Assumption1: overestimated speed v = 1.5c 
    """
    F1, Cov_w1 = build_F_Cov(1.5*c, sn, gamma1, 1)
    F2, Cov_w2 = build_F_Cov(c, sn, gamma2, 1)
   
    Cov_k =  np.eye(sn+1)
    Mean_k = np.zeros(sn+1)

    X1 = Kalmen_Filer(Y, G, F1, tn, Cov_w1, Cov_k, Mean_k)
    X2 = Kalmen_Filer(Y, G, F2, tn, Cov_w2, Cov_k, Mean_k)
    
    t = np.linspace(1,20,tn).reshape((1,tn))
    Space, Time = np.meshgrid(s,t)

    """
        Assumption 1: speed = c
    """
    figcont += 1
    fig = plt.figure(figcont, figsize=(8,4)) 
    ax = Axes3D(fig)
    surf = ax.plot_surface(Space, Time, X1, facecolors=cm.Blues_r(X1/4*X1.max()),
                           antialiased=False,shade =False)
    ax.view_init(elev = 40, azim= 235)
    ax.set_zlim(-2,2)
    ax.set_zticks(np.array([-2,0,2]))
    ax.grid(False)
    plt.xlim(-0.05,1)
    plt.ylim(-0.5,20)
    plt.xlabel("space(s)",size = 12)
    plt.ylabel("time(t)",size = 12)   
    plt.xticks(np.linspace(0,1,6))
    plt.yticks(np.linspace(0,tn,tn//5+1))
    plt.tick_params(labelsize='10')
    plt.title("wave propagation model\n(overestimated speed)", fontsize = 15)
    plt.grid()
    #plt.savefig("chap4_ex2_"+str(figcont)+".eps")
    plt.show()
    
    """
        Assumption 1: speed = c
    """
    figcont += 1
    fig = plt.figure(figcont, figsize=(8,4)) 
    ax = Axes3D(fig)
    surf = ax.plot_surface(Space, Time, X2, facecolors=cm.Blues_r(X2/4*X2.max()),
                           antialiased=False,shade =False)
    ax.view_init(elev = 40, azim= 235)
    ax.set_zlim(-2,2)
    ax.set_zticks(np.array([-2,0,2]))
    ax.grid(False)
    plt.xlim(-0.05,1)
    plt.ylim(-0.5,20)
    plt.xlabel("space(s)",size = 12)
    plt.ylabel("time(t)",size = 12)   
    plt.xticks(np.linspace(0,1,6))
    plt.yticks(np.linspace(0,tn,tn//5+1))
    plt.tick_params(labelsize='10')
    plt.title("wave propagation model\n(correct speed)", fontsize = 15)
    plt.grid()
    #plt.savefig("chap4_ex2_"+str(figcont)+".eps")
    plt.show()
    
   