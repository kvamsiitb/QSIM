#!/usr/bin/env python
# coding: utf-8

# In[1]:


### IMPORTING REQUISITE PACKAGES
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'D://Intel//SPIM//SLM-code//SLM')


import numpy as np
import time
import copy
import matplotlib
import matplotlib.pyplot as plt

import csv
import os
from os.path import isfile, join
import random
from random import randint
import cv2


#import scipy as sp
#from scipy.signal import *



from pypylon import pylon
from pypylon import genicam
import sys  # We need sys so that we can pass argv to QApplication


import detect_heds_module_path
from holoeye import slmdisplaysdk

imgx1 = 680
imgx2 = 731
imgy1 = 529
imgy2 = 586
'''
input: ( (512, 512), (8, 8) )
for i in range(0, 64):
 ...

arr = [(0,0), (0, 1), (0, 2), (0, 3),.. , (63, 63)]
'''
def spin_tuple(shape, bin_size):
    arr = []
    '''
    x = np.linspace(0, shape[0]//bin_size-1, shape[0]//bin_size)
     
    # numpy.linspace creates an array of
    # 9 linearly placed elements between
    # -4 and 4, both inclusive
    y = np.linspace(0, shape[1]//bin_size-1, shape[1]//bin_size)
    xx, yy = np.meshgrid(x, y, indexing = 'ij')        
    '''
    for i in range(0, shape[0]//bin_size):
        for j in range(0, shape[1]//bin_size):
            arr.append((i,j))
            '''
            if ( (i, j) == ( int(xx[i, j]), int(yy[i, j]))  ):
                print('q ', end = ' ' )
            else:
                print('sadasadas')
            '''            

    return arr
'''
pattern generation for

area = (1024, 1024)
bin = (16, 16)
blocks of 128 or 0
'''    
def checkerboard(shape, bin):
    phase = np.zeros(shape)
    for i in range(shape[0]//bin):
        for j in range(shape[1]//bin):
            phase[i*bin: i*bin + bin, j*bin: j*bin + bin] = 128*((i+j)%2)  
    return phase
 
def flip_np(spin_arr, x, tot_shape, shape, bin, d):
    y = copy.copy(x)
    l = random.sample(spin_arr,d)
    for i in range(d):
        y[(tot_shape[0]-shape[0])//(2)+l[i][0]*bin:(tot_shape[0]-shape[0])//(2)+l[i][0]*bin + bin, (tot_shape[1]-shape[1])//(2)+l[i][1]*bin:(tot_shape[1]-shape[1])//(2)+l[i][1]*bin + bin] = (np.pi + y[(tot_shape[0]-shape[0])//(2)+l[i][0]*bin:(tot_shape[0]-shape[0])//(2)+l[i][0]*bin + bin, (tot_shape[1]-shape[1])//(2)+l[i][1]*bin:(tot_shape[1]-shape[1])//(2) + l[i][1]*bin + bin])%(2*np.pi)
    return y
    
bins = 2**3 # 8
outer_bins = 2**4 # 16

# check here 
beta = np.arange(600,0,-0.2)
count=0
active_area = (2**9,2**9) # (512, 512)
spin_arr = spin_tuple(active_area, bins)
area = (2**10,2**10) # (1024, 1024)
mask = np.zeros(area)
d = 2**2

loss_arr = [0]
spinflip = []
fidelity_arr = []

# Initializes the SLM library
slm = slmdisplaysdk.SLMInstance()

# Check if the library implements the required version
if not slm.requiresVersion(3):
    exit(1)

# Detect SLMs and open a window on the selected SLM
error = slm.open()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Open the SLM preview window in "Fit" mode:
# Please adapt the file showSLMPreview.py if preview window
# is not at the right position or even not visible.
from showSLMPreview import showSLMPreview
#showSLMPreview(slm, scale=0.0)


# Reserve memory for the data:
dataWidth = slm.width_px
dataHeight = slm.height_px

x = slmdisplaysdk.createFieldUChar(area[0],area[1])
 
  
number_part= np.loadtxt('numbers.csv', delimiter=",") # = np.ones((64, 64)) #np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
x = np.pi*checkerboard(area,outer_bins)/128 # 0 or 3.14
'''
4//2
a[i*2: (i + 1)*2, j*2: (j + 1)*2 ] = (i+j)%2

temp = array([[0., 0., 1., 1.],
               [0., 0., 1., 1.],
               [1., 1., 0., 0.],
               [1., 1., 0., 0.]])
'''
temp = np.zeros((bins*number_part.shape[0],bins*number_part.shape[1]))

for i in range(number_part.shape[0]):
    for j in range(number_part.shape[1]):
        temp[i*bins:i*bins+bins, j*bins:j*bins+bins] = (i+j)%2#number_part[i,j]

temp = np.pi * temp

x[(area[0]-active_area[0])//2:(area[0]+active_area[0])//2, (area[1]-active_area[1])//2:(area[1]+active_area[1])//2] = temp


error = slm.showPhasevalues(x)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
time.sleep(0.15)

plt.imshow(x)
plt.show()

imgx1 = input("imgx1: ")
imgy1 = input("imgy1: ")
imgx2 = input("imgx2: ")
imgy2 = input("imgy2: ")

'''

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.ExposureTime.SetValue(45)

while not (count == len(beta) - 1):
    print("%%% ==> %%^%")
    x2 = flip_np(spin_arr, x, area, active_area, bins, d)

    error = slm.showPhasevalues(x2+mask)
    #assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
    print('w', end = ' ')
    time.sleep(0.05) 
    
    camera.StartGrabbingMax(1)

    while camera.IsGrabbing():
    # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            # Access the image data.
            #img = rebin(grabResult.Array, imgbin)
            img = (grabResult.Array)[imgx1:imgx2,imgy1:imgy2]

        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        grabResult.Release()
    

    C1 = np.sum(img)/np.size(img)

    delE = C1 - loss_arr[-1]
    #print(C1, COST[-1], delE)

    p = np.exp(-1*beta[count]*delE)                


    if delE <= 0:
        x=x2
        #print("Accepted")
        loss_arr.append(C1)
        spinflip.append(d)
    elif np.random.choice(a=[0,1], p=[p, 1-p])==0:
        x=x2
        loss_arr.append(C1)
        spinflip.append(d)
    else:
        x=x
        loss_arr.append(loss_arr[-1])
        spinflip.append(0)
        
    count += 1
    final_screen = x[(area[0]-active_area[0])//2:(area[0]+active_area[0])//2, (area[1]-active_area[1])//2:(area[1]+active_area[1])//2]


    s1 = np.where(final_screen < np.pi)
    s2 = np.where(final_screen >= np.pi)
    fidelity = (np.sum(number_part[s1])-np.sum(number_part[s2]))/(np.sum(number_part[s1])+np.sum(number_part[s2]))
    fidelity_arr.append(fidelity)

print(loss_arr)

# CLOSING DOWN THE INSTRUMENTS

camera.Close()

'''


'''

'''
val = input("Enter your value after ur done: ")
print(val)
# Wait until the SLM process is closed:
print("Waiting for SDK process to close. Please close the tray icon to continue ...")
error = slm.utilsWaitUntilClosed()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Unloading the SDK may or may not be required depending on your IDE:
slm = None

# SHOWING ALL THE PLOTS GENERATED
#plt.show()

