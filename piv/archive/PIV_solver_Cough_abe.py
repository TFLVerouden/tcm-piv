# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:33:27 2024

@author: thijs
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 
from scipy import signal
from scipy import datasets
import glob
# path = r"D:\Experiments\PIV\250624_1323_80ms_3"
path = '/Volumes/Data/Data/250623 PIV/250624_1115_thirdtest'
file_names = glob.glob(f"{path}\\*.tif")
file_names = file_names[500:505]

#%%
def corr(img1, img2):
    corr = signal.correlate2d(img2,img1,mode='same')
    if np.max(corr)>0.0:
        max0,max1 = np.unravel_index(np.argmax(corr), corr.shape)
        submax0 = max0 + subpix(max0,max1,corr,'0') + 0.5 # correction due to center of pixel vs center of image
        submax1 = max1 + subpix(max0,max1,corr,'1') + 0.5
        vectorx = submax1-(nx-1)/2 # axis 1 correspond with x displacement
        vectory = submax0-(ny-1)/2 # axis 0 correspond with y displacement
    else:
        vectorx,vectory = 0,0
    return vectorx,vectory
    
def subpix(max0,max1,corr,axis):
    i = corr[max0,max1]
    try:
        if axis == '0':
            ip = corr[max0+1,max1]
            im = corr[max0-1,max1]
        else:
            ip = corr[max0,max1+1]
            im = corr[max0,max1-1]
        if ip>0.0 and im>0.0 and i>0.0:
            if (np.log(ip)+np.log(im)-2*np.log(i)) != 0:
                if abs(1/2 * (np.log(im) - np.log(ip)) / (np.log(ip)+np.log(im)-2*np.log(i)))<1.0:
                    return 1/2 * (np.log(im) - np.log(ip)) / (np.log(ip)+np.log(im)-2*np.log(i))
        return 0.0
    except: 
        return 0.0
        
def PIV(filea,fileb,nx,ny,N0,N1):

    img_a = cv2.imread(filea,cv2.IMREAD_GRAYSCALE).astype(float)
    img_b = cv2.imread(fileb,cv2.IMREAD_GRAYSCALE).astype(float)
    
    img_a = img_a/255
    img_b = img_b /255

    vector = np.zeros((2,N0,N1))
    for i in range(N0):
        for j in range(N1):
            w_a = img_a[i*nx:(i+1)*nx,j*ny:(j+1)*ny]
            w_b = img_b[i*nx:(i+1)*nx,j*ny:(j+1)*ny]
            vector[:,i,j] = corr(w_a,w_b) 
    return vector

#%% 
img_1 = cv2.imread(file_names[0],cv2.IMREAD_GRAYSCALE).astype(float)/255
Nfiles = len(file_names)
nx = 16
ny =1
N0 = 832//nx
N1 = 384//ny
X,Y = np.meshgrid(np.arange((ny-1)/2,ny*N1,ny), np.arange((nx-1)/2,nx*N0,nx))
dx = np.zeros((Nfiles-1,N0,N1))
dy = np.zeros((Nfiles-1,N0,N1))
#%% 
for idx in range(1):
    vector = PIV(file_names[idx],file_names[idx+1],nx,ny,N0,N1)
    dx[idx] =  vector[0]
    dy[idx] =  vector[1] 
    print(idx/(Nfiles-1))
np.savez('cough_test', dx=dx, dy=dy)



#%%
dx_dy = np.load(r"cough_test.npz")

dx = dx_dy['dx'][:]
dy = dx_dy['dy'][:]

#%% Average
dx_avg = np.sum(dx,axis=0)/len(dx)
dy_avg = np.sum(dy,axis=0)/len(dy)
print(dx_avg)
print(dy_avg)
print(np.shape(dx_avg))
print(np.shape(dy_avg))

#%% Quiver
plt.figure()
plt.imshow(img_1,cmap='grey')
plt.quiver(X,Y,dx[0],dy[0],angles='xy', scale_units='xy',scale=0.05, color='green')
plt.show()

  #%% Countour
velocity = np.sqrt(dx_avg**2+dy_avg**2)
velocity1 = np.sqrt(dx[0]**2+dy[0]**2)
plt.figure()
plt.contourf(X,Y,velocity1,levels=np.linspace(0,5,101),cmap='RdBu_r')
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()

#%% Histogram
plt.figure()
plt.hist(dy_avg.ravel(), bins=100,density=True)
plt.xlabel(r'$\Delta$Y')
plt.ylabel('Count')
plt.figure()
plt.hist(dx_avg.ravel(), bins=100,density=True)
plt.xlabel(r'$\Delta$X')
plt.ylabel('Count')
plt.show()

#%% Scatter
plt.figure()
plt.scatter(dx[0],-dy[0])
plt.axhline(y=0,ls='--')
plt.axvline(x=0,ls='--')
plt.xlabel(r'$\Delta$X')
plt.ylabel(r'$\Delta$Y')
plt.show()

#%% Velocity height
scale = 100
plt.figure()
plt.imshow(img_1,cmap='grey')
plt.quiver(X,Y,dx[0],dy[0],angles='xy', scale_units='xy',scale=0.01, color='green')
plt.plot(np.arange((n-1)/2,n*N1,n),n*95+scale*np.mean(dy_avg[90:100],axis=0),'--o',label='0-5')
plt.plot(np.arange((n-1)/2,n*N1,n),n*15+scale*np.mean(dy_avg[10:20],axis=0),'--o',label='5-10')
plt.plot(np.arange((n-1)/2,n*N1,n),n*25+scale*np.mean(dy_avg[20:30],axis=0),'--o',label='10-15')
plt.plot(np.arange((n-1)/2,n*N1,n),n*35+scale*np.mean(dy_avg[30:40],axis=0),'--o',label='15-20')
plt.plot(np.arange((n-1)/2,n*N1,n),n*45+scale*np.mean(dy_avg[40:50],axis=0),'--o',label='20-25')
plt.plot(np.arange((n-1)/2,n*N1,n),n*55+scale*np.mean(dy_avg[50:60],axis=0),'--o',label='25-30')
plt.plot(np.arange((n-1)/2,n*N1,n),n*85+scale*np.mean(dy_avg[80:90],axis=0),'--o',label='25-30')
plt.plot(np.arange((n-1)/2,n*N1,n),n*115+scale*np.mean(dy_avg[110:120],axis=0),'--o',label='25-30')
plt.xlabel('pix')
plt.ylabel(r'$\Delta$Y')
#plt.legend(frameon=True)

