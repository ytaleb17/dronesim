# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:39:50 2015

@author: droneswarm
"""
import numpy as np
from numpy import pi, sin, cos, sum, dot, max, sqrt, arctan2, arccos
import numpy.random as nprand
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time as Time
import struct

#from geopy.geocoders import Nominatim
#geolocator = Nominatim()
#location = geolocator.geocode("4500 Forbes Blvd Lanham MD")

def lla2ecef(lla):
    lat = lla[0][0]*pi/180.0
    lon = lla[1][0]*pi/180.0
    alt = lla[2][0] # m
    
    # WGS84 ellipsoid constants:
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1.0-f)
    e = sqrt((a**2 - b**2)/a**2)
    
    N = a / sqrt(1.0 - e**2 * sin(lat)**2)
    
    x = (N+alt) * cos(lat) * cos(lon)
    y = (N+alt) * cos(lat) * sin(lon)
    z = ((1.0-e**2) * N + alt) * sin(lat)
    return np.array([[x],[y],[z]])

def ecef2lla(pos):
    x = pos[0][0] #m
    y = pos[1][0] #m
    z = pos[2][0] #m
    
    longi = arctan2(y,x)*180.0/pi
    p = sqrt(x**2 + y**2)
    
    # WGS84 ellipsoid constants:
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1.0-f)
    e = sqrt((a**2 - b**2)/a**2)
    ep = sqrt((a**2 - b**2)/b**2)
    
    th = arctan2(z*a,p*b)
    lat = arctan2((z + ep**2 * b * sin(th)**3),(p - e**2 * a * cos(th)**3))
    N = a / sqrt(1.0 - e**2 * sin(lat)**2)
    h = (p/cos(lat)) - N
    lat = lat*180.0/pi
    return np.array([[lat],[longi],[h]])

def enu2ecef(enu, lat, lon, ref_ecef):
    lat = lat*pi/180.0
    lon = lon*pi/180.0
    
    R = np.array([[-sin(lon), -sin(lat)*cos(lon), cos(lat)*cos(lon)],
                  [cos(lon), -sin(lat)*sin(lon), cos(lat)*sin(lon)],
                  [0.0, cos(lat), sin(lat)]])
    ned = dot(R,enu) + ref_ecef
    return ned
    
    
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
    
#    R = np.array([[cos(ph)*cos(ps) - cos(th)*sin(ph)*sin(ps), -cos(ps)*sin(ph) - cos(ph)*cos(th)*sin(ps), sin(th)*sin(ps)],
#                   [cos(th)*cos(ps)*sin(ph) + cos(ph)*sin(ps), cos(ph)*cos(th)*cos(ps) - sin(ph)*sin(ps), -cos(ps)*sin(th)],
#                   [sin(ph)*sin(th), cos(ph)*sin(th), cos(th)]]) 
                   
    R = dot(rotz(ps),dot(roty(th),rotx(ph)))
    return R

def rotx(angle):
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cos(angle), -sin(angle)],
                   [0.0, sin(angle), cos(angle)]])
    
    return Rx
 
def roty(angle):
    Ry = np.array([[cos(angle), 0.0, sin(angle)],
                   [0.0,        1.0,        0.0],
                   [-sin(angle), 0.0, cos(angle)]])
    
    return Ry
              
def rotz(angle):
    Rz = np.array([[cos(angle), -sin(angle), 0.0],
                   [sin(angle),  cos(angle), 0.0],
                   [0.0, 0.0, 1.0]])
    
    return Rz
    
def thrust(inputs,k):
    T_M = k*inputs
    b = 1.0/sqrt(2.0)
    alpha = 90.0-8.0
    sa = sin(alpha)
    ca = cos(alpha)
    e_M = np.vstack(([0.0,-sa,-ca],
                     [b*sa,b*sa,-ca],
                     [-sa,0.0,-ca],
                     [b*sa,-b*sa,-ca],
                     [0.0, sa,-ca],
                     [-b*sa,-b*sa,-ca],
                     [sa,0.0,-ca],
                     [-b*sa,b*sa,-ca])).transpose()
                     
    T = np.dot(e_M,T_M)                 
    
    return T

def torques(inputs, L, b, k):
    T_M = k*inputs
    b = 1.0/sqrt(2.0)
    r_M = np.vstack(([L, 0.0, 0.0],
                     [b*L, -b*L, 0.0],
                     [0.0, -L, 0.0],
                     [-b*L, -b*L, 0.0],
                     [-L, 0.0, 0.0],
                     [-b*L, b*L, 0.0],
                     [0.0, L, 0.0],
                     [b*L, b*L, 0.0])).transpose()
                     
    alpha = 90.0-8.0
    sa = sin(alpha)
    ca = cos(alpha)
    e_M = np.vstack(([0.0,-sa,-ca],
                     [b*sa,b*sa,-ca],
                     [-sa,0.0,-ca],
                     [b*sa,-b*sa,-ca],
                     [0.0, sa,-ca],
                     [-b*sa,-b*sa,-ca],
                     [sa,0.0,-ca],
                     [-b*sa,b*sa,-ca])).transpose()    
    
    tau = np.zeros((3,1))
    for i in range(0,8):  
        tau = tau + np.cross(r_M[:,i], e_M[:,i]*T_M[i][0]).reshape(3,1) + (((-1)**i)*e_M[:,i]*b*inputs[i][0]).reshape(3,1)
        
    return tau

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])
    
def double_to_hex(f):
    return hex(struct.unpack('<Q', struct.pack('<d', f))[0])

def angdif(v1,v2):
    angle = arccos(dot(v1,v2)/(norm(v1)*norm(v2)))
    return angle
    
# time
t0 = 0
tf = 10
dt = 0.005
t =  np.arange(t0,tf,dt)

# initial states
x = np.array([[0.0],[0.0],[10.0]])
xd = np.zeros((3,1))
theta = np.zeros((3,1))
deviation = 100
thetad = np.deg2rad(2*deviation*nprand.rand(3,1) - deviation)
#thetad = np.zeros((3,1))
inputs = np.zeros((8,1))

# physical parameters
g = 9.81 # m/s^2
m = 0.468 # kg
L = (378.0+(337.5/2))*1e-3 #m
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


bat_volt = np.zeros((1,len(t)))
can_volt = np.zeros((1,len(t)))
bec_volt = np.zeros((1,len(t)))
speed = np.zeros((1,len(t)))
numSat = np.zeros((1,len(t)), dtype=np.uint8)
date = np.zeros((1,len(t)), dtype=np.uint32)
time = np.zeros((1,len(t)), dtype=np.uint32)
year = np.zeros((1,len(t)), dtype=np.uint8)
month = np.zeros((1,len(t)), dtype=np.uint8)
day = np.zeros((1,len(t)), dtype=np.uint8)
hour = np.zeros((1,len(t)), dtype=np.uint8)
minute = np.zeros((1,len(t)), dtype=np.uint8)
second = np.zeros((1,len(t)), dtype=np.uint8)
mag = np.zeros((3,len(t)), dtype=np.int16)
lla = np.zeros((3,len(t)))
# reference lat long ai solutions
lat = 38.983381 #deg
longi = -76.8270829 #deg
reflla = np.array([[lat],[longi],[0.0]])
refpos_ecef = lla2ecef(reflla)
#lat = location.latitude #deg
#longi = location.longitude # deg 

sample = 0
f2hex = np.vectorize(float_to_hex,otypes=[object])
d2hex = np.vectorize(double_to_hex,otypes=[object])
i2hex = np.vectorize(hex,otypes=[object])
#bytetable = np.empty([54,len(t)],dtype=object)
bytepackets = bytearray()
startpacket = '7E7E7E'.decode('hex')
endpacket = '7F7F7F'.decode('hex')
for i in range(len(t)):
    
    bytepackets.extend(startpacket)
    tlocal = Time.localtime()
    date[0][i] = tlocal.tm_year*10000 + tlocal.tm_mon*100+tlocal.tm_mday
    time[0][i] = tlocal.tm_hour*10000 + tlocal.tm_min*100 + tlocal.tm_sec
    year[0][i] = tlocal.tm_year-2000
    month[0][i] = tlocal.tm_mon
    day[0][i] = tlocal.tm_mday
    hour[0][i] = tlocal.tm_hour
    minute[0][i] = tlocal.tm_min
    second[0][i] = tlocal.tm_sec
    
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
    
    # check if hit floor
    if x[2][0] <= 0.0:
        x[2][0] = 0.0
        xd = np.zeros((3,1))
        thetad = np.zeros((3,1))
        w = np.zeros((3,1))
        
    pos[:,i] = x.ravel()
    vel[:,i] = xd.ravel()
    accel[:,i] = a.ravel()
    att[:,i] = theta.ravel()
    
    speed[0][i] = norm(xd)
    mag[:,i] = nprand.randint(-200,200,[1,3])
    
    lla[:,i] = ecef2lla(enu2ecef(x,lat,longi,refpos_ecef)).ravel()
    
    # Heading calculations
    mh = dot(roty(theta[1][0]),dot(rotx(theta[0][0]),np.array([[mag[0,i]],[mag[1,i]],[mag[2,i]]])))
    head_nc = arctan2(mag[1,i],mag[0,i])*180.0/pi
    head_c = arctan2(mh[1][0],mh[0][0])*180.0/pi
    head_c_x = np.float32(mh[0][0])
    head_c_y = np.float32(mh[1][0])
    
    if i == 0:
        lla_dif = np.array([0.0,0.0,0.0])
    else:
        lla_dif = lla[:,i] - lla[:,i-1]
    
    vsi = xd[2][0]
    vsi_gps = xd[2][0]
    cog = angdif(np.array([1.0,0.0]),lla_dif[0:2])*180/pi
    velocity_n_gps = xd[0][0]*1e2
    velocity_e_gps = xd[1][0]*1e2
    velocity_d_gps  = -xd[2][0]*1e2   
    velocity_n = xd[0][0]*1e2
    velocity_e = xd[1][0]*1e2
    velocity_d  = -xd[2][0]*1e2   
    
    if sample >= 1.0:
        # Placeholder values
        bat_volt[0][i] = (22.2 + nprand.uniform(-1,1))*1e3
        can_volt[0][i] = (5.0 + nprand.uniform(-0.5,0.5))*1e3
        bec_volt[0][i] = (5.0 + nprand.uniform(-0.5,0.5))*1e3
        numSat[0][i] = nprand.randint(3,12)
        sample = 0.0
    else:
        sample += dt
        if i == 0:
            bat_volt[0][i] = (22.2 + nprand.uniform(-1,1))*1e3
            can_volt[0][i] = (5.0 + nprand.uniform(-0.5,0.5))*1e3
            bec_volt[0][i] = (5.0 + nprand.uniform(-0.5,0.5))*1e3
            numSat[0][i] = nprand.randint(3,12)
        else:  
            bat_volt[0][i] = bat_volt[0][i-1]
            can_volt[0][i] = can_volt[0][i-1]
            bec_volt[0][i] = bec_volt[0][i-1]
            numSat[0][i] = numSat[0][i-1]

    bytepackets.extend('00'.decode('hex'))
    bytepackets.extend(numSat[0][i].tobytes())
    bytepackets.extend('01'.decode('hex'))
    bytepackets.extend(date[0][i].tobytes())
    bytepackets.extend('02'.decode('hex'))
    bytepackets.extend(time[0][i].tobytes())
    bytepackets.extend('03'.decode('hex'))
    bytepackets.extend(year[0][i].tobytes())
    bytepackets.extend('04'.decode('hex'))
    bytepackets.extend(month[0][i].tobytes())
    bytepackets.extend('05'.decode('hex'))
    bytepackets.extend(day[0][i].tobytes())
    bytepackets.extend('06'.decode('hex'))
    bytepackets.extend(hour[0][i].tobytes())
    bytepackets.extend('07'.decode('hex'))
    bytepackets.extend(minute[0][i].tobytes())
    bytepackets.extend('08'.decode('hex'))
    bytepackets.extend(second[0][i].tobytes())
    bytepackets.extend('09'.decode('hex'))
    bytepackets.extend(a[0].astype(np.float32).tobytes())
    bytepackets.extend('0A'.decode('hex'))
    bytepackets.extend(a[1].astype(np.float32).tobytes())
    bytepackets.extend('0B'.decode('hex'))
    bytepackets.extend(a[2].astype(np.float32).tobytes())
    bytepackets.extend('0C'.decode('hex'))
    bytepackets.extend(mag[0][i].tobytes())
    bytepackets.extend('0D'.decode('hex'))
    bytepackets.extend(mag[1][i].tobytes())
    bytepackets.extend('0E'.decode('hex'))
    bytepackets.extend(mag[2][i].tobytes())
    bytepackets.extend('0F'.decode('hex'))
    bytepackets.extend((lla[0,i]*pi/180).astype(np.float64).tobytes())
    bytepackets.extend('10'.decode('hex'))
    bytepackets.extend((lla[1,i]*pi/180).astype(np.float64).tobytes())
    bytepackets.extend('11'.decode('hex'))
    bytepackets.extend((lla[0,i]*1e7).astype(np.uint32).tobytes())
    bytepackets.extend('12'.decode('hex'))
    bytepackets.extend((lla[1,i]*1e7).astype(np.uint32).tobytes())
    bytepackets.extend('13'.decode('hex'))
    bytepackets.extend((x[2][0]).astype(np.float32).tobytes())
    bytepackets.extend('14'.decode('hex'))
    bytepackets.extend((lla[2,i]*1e3).astype(np.uint32).tobytes())
    bytepackets.extend('15'.decode('hex'))
    bytepackets.extend(head_c.tobytes())
    bytepackets.extend('16'.decode('hex'))
    bytepackets.extend(head_nc.tobytes())
    bytepackets.extend('17'.decode('hex'))
    bytepackets.extend(head_c_x.tobytes())
    bytepackets.extend('18'.decode('hex'))
    bytepackets.extend(head_c_y.tobytes())
    bytepackets.extend('19'.decode('hex'))
    bytepackets.extend(speed[0][i].tobytes())
    bytepackets.extend('1A'.decode('hex'))
    bytepackets.extend(vsi.tobytes())
    bytepackets.extend('1B'.decode('hex'))
    bytepackets.extend(vsi_gps.tobytes())
    bytepackets.extend('1C'.decode('hex'))
    bytepackets.extend(cog.tobytes())
    bytepackets.extend('1D'.decode('hex'))
    bytepackets.extend(velocity_n_gps.astype(np.float32).tobytes())
    bytepackets.extend('1E'.decode('hex'))
    bytepackets.extend(velocity_e_gps.astype(np.float32).tobytes())
    bytepackets.extend('1F'.decode('hex'))
    bytepackets.extend(velocity_d_gps.astype(np.float32).tobytes())
    bytepackets.extend('20'.decode('hex'))
    bytepackets.extend(velocity_n.astype(np.float32).tobytes())
    bytepackets.extend('21'.decode('hex'))
    bytepackets.extend(velocity_e.astype(np.float32).tobytes())
    bytepackets.extend('22'.decode('hex'))
    bytepackets.extend(velocity_d.astype(np.float32).tobytes())
    bytepackets.extend('23'.decode('hex'))
    bytepackets.extend(bat_volt[0][i].astype(np.uint16).tobytes())
    bytepackets.extend('24'.decode('hex'))
    bytepackets.extend(can_volt[0][i].astype(np.uint16).tobytes())
    bytepackets.extend('25'.decode('hex'))
    bytepackets.extend(bec_volt[0][i].astype(np.uint16).tobytes())
    
    bytepackets.extend(endpacket)
    
plt.close()    
plt.figure(1)
plt.subplot(411)
plt.plot(t,pos[0,:],'r',t,pos[1,:],'b',t,pos[2,:],'g')

plt.subplot(412)
plt.plot(t,vel[0,:],'r',t,vel[1,:],'b',t,vel[2,:],'g')

plt.subplot(413)
plt.plot(t,accel[0,:],'r',t,accel[1,:],'b',t,accel[2,:],'g')

plt.subplot(414)
plt.plot(t,att[0,:],'r',t,att[1,:],'b',t,att[2,:],'g')
plt.show()