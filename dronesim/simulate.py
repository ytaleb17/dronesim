# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:39:50 2015

@author: droneswarm
"""
import numpy as np
from numpy import sin, cos, sum, dot, max
from numpy.random import rand
from numpy.linalg import inv
import matplotlib.pyplot as plt

def thetadot2omega(thetad,theta):
    ph = theta[0]
    th = theta[1]
    ps = theta[2]
    R = np.array([[1, 0, -sin(th)],
                   [0, cos(ph), cos(ph)*sin(ph)],
                   [0, -sin(ph), cos(th)*cos(ph)]])
    w = dot(R,thetad)
    
    return w
    
def omega2thetadot(w,theta):
    ph = theta[0]
    th = theta[1]
    ps = theta[2]
    
    R = np.array([[1, 0, -sin(th)],
                   [0, cos(ph), cos(ph)*sin(ph)],
                   [0, -sin(ph), cos(th)*cos(ph)]])
                   
    thetad = dot(inv(R),w)
    
    return thetad
    
def lin_accel(inputs, angles, xd, m, g, k, kdrag):
    gravity = np.array([[0],[0],[-g]])
    R = rotation(angles)
    T = dot(R,thrust(inputs,k))
    Fdrag = -dot(kdrag,xd)
    a = gravity + (1/m) * T + Fdrag
    
    return a
    
def ang_accel(inputs, w, I, L, b, k):
    tau = torques(inputs, L, b, k)
    wd = dot(inv(I),(tau - np.cross(w.transpose(), dot(I,w).transpose()).transpose()))
    
    return wd
    
def rotation(theta):
    # Body to Inertial Rotation
    ph = theta[0][0]
    th = theta[1][0]
    ps = theta[2][0]
    
    R = np.array([[cos(ph)*cos(ps) - cos(th)*sin(ph)*sin(ps), -cos(ps)*sin(ph) - cos(ph)*cos(th)*sin(ps), sin(th)*sin(ps)],
                   [cos(th)*cos(ps)*sin(ph) + cos(ph)*sin(ps), cos(ph)*cos(th)*cos(ps) - sin(ph)*sin(ps), -cos(ps)*sin(th)],
                   [sin(ph)*sin(th), cos(ph)*sin(th), cos(th)]]) 
    return R

def thrust(inputs,k):
    T = np.array([[0],
                  [0],
                  [k*sum(inputs)]])
    
    return T

def torques(inputs, L, b, k):
    tau = np.array([L*k*(inputs[0] - inputs[2]),
                    L*k*(inputs[1] - inputs[3]),
                    b*(inputs[0] - inputs[1] + inputs[2] - inputs[3])])
    return tau
    
# time
t0 = 0
tf = 10
dt = 0.005
t =  np.arange(t0,tf,dt)

# initial states
x = np.array([[0],[0],[10]])
xd = np.zeros((3,1))
theta = np.zeros((3,1))
deviation = 100
#thetad = np.deg2rad(2*deviation*rand(3,1) - deviation)
thetad = np.zeros((3,1))
inputs = np.zeros((4,1))

# physical parameters
g = 9.81 # m/s^2
m = 0.468 # kg
L = 0.225 # m
k = 2.98e-6
b = 1.14e-7
Im = 3.357e-5
Ixx = 4.856e-3 # kg m^2
Iyy = 4.856e-3 # kg m^2
Izz = 8.801e-3 # kg m^2
Ax = 0.25 # kg/s
Ay = 0.25 # kg/s
Az = 0.25 # kg/s

I = np.array([[Ixx,0,0],
              [0,Iyy,0],
              [0,0,Izz]])
              
kdrag = np.array([[Ax,0,0],
                  [0,Ay,0],
                  [0,0,Az]])      
                  
pos = np.zeros((3,len(t)))
vel = np.zeros((3,len(t)))
accel = np.zeros((3,len(t)))
att = np.zeros((3,len(t)))

for i in range(len(t)):
    
    ti = t[i]
    # compute angular velocity
    w = thetadot2omega(thetad,theta)
    
    # compute linear and angular accelerations
    a = lin_accel(inputs, theta, xd, m, g, k, kdrag)
    wd = ang_accel(inputs, w, I, L, b, k)
    
    w += dt*wd
    thetad = omega2thetadot(w, theta)
    
    theta += dt*thetad
    xd += dt*a
    x += dt*xd
  
    #x = max((np.array([[0,0,0]]).transpose(),x),0)
    
    pos[:,i] = x.transpose()
    vel[:,i] = xd.transpose()
    accel[:,i] = a.transpose()
    att[:,i] = theta.transpose()

plt.close()    
#plt.figure(1)
#plt.subplot(411)
#plt.plot(t,pos[0,:],'r',t,pos[1,:],'b',t,pos[2,:],'g')
#
#plt.subplot(412)
#plt.plot(t,vel[0,:],'r',t,vel[1,:],'b',t,vel[2,:],'g')
#
#plt.subplot(413)
#plt.plot(t,accel[0,:],'r',t,accel[1,:],'b',t,accel[2,:],'g')
#
#plt.subplot(414)
#plt.plot(t,att[0,:],'r',t,att[1,:],'b',t,att[2,:],'g')

plt.show()