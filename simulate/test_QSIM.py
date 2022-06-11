import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import time

# https://www.youtube.com/watch?v=jclknhNJBrE&ab_channel=SteveBrunton
# https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/
### IMPORTING REQUISITE PACKAGES
import sys
# insert at 1, 0 is the script path (or '' in REPL)


imgx1 = 810#852
imgx2 = 841#876
imgy1 = 450#580
imgy2 = 487#623

'''
Def: Generate meshgrid given by 
      x = np.linspace(0, shape[0]//bin_size-1, shape[0]//bin_size)
      y = np.linspace(0, shape[1]//bin_size-1, shape[1]//bin_size)

output: xx, yy = np.meshgrid(x, y, indexing = 'ij')    

smallest ex [(0,0)...,(3, 7)] given spins are 32 spins stacked as 4 rows N 8 columns    
'''
def spin_tuple(shape, bin_size):
    arr = []
    for i in range(0, shape[0]//bin_size):
        for j in range(0, shape[1]//bin_size):
            arr.append((i,j))
    return arr

'''
Def: pattern generation for
    area = (1024, 1024)
    bin = (16, 16)

output: blocks of 128 or 0
'''    
def checkerboard(shape, bin):
    phase = np.zeros(shape)
    for i in range(shape[0]//bin):
        for j in range(shape[1]//bin):
            phase[i*bin: i*bin + bin, j*bin: j*bin + bin] = 128*((i+j)%2)  
    return phase

'''
Def: decide where to flip from 0/pi or pi/2 / 3pi/2
output: 1(makes spin pi) or -1(makes spin 0)

>>> (1+1)%2
0
>>> (1+0)%2
1
>>> (1+3/2)%2
0.5
>>> (1+1/2)%2
1.5
'''
def flip_np1(spin_arr, x, tot_shape, shape, bin, d):
    y = copy.copy(x)
    l = random.sample(spin_arr,d)
    for i in range(d):
        y[(tot_shape[0]-shape[0])//(2)+l[i][0]*bin:(tot_shape[0]-shape[0])//(2)+l[i][0]*bin + bin, (tot_shape[1]-shape[1])//(2)+l[i][1]*bin:(tot_shape[1]-shape[1])//(2)+l[i][1]*bin + bin] = (np.pi + y[(tot_shape[0]-shape[0])//(2)+l[i][0]*bin:(tot_shape[0]-shape[0])//(2)+l[i][0]*bin + bin, (tot_shape[1]-shape[1])//(2)+l[i][1]*bin:(tot_shape[1]-shape[1])//(2) + l[i][1]*bin + bin])%(2*np.pi)
    return y

#each_IsingShape (256,512 )
def flip_np2(spin_arr, x, tot_shape, shape, bin, d, each_IsingShape):
    y = copy.copy(x)
    l = random.sample(spin_arr,d)
    for i in range(d):
      y[(tot_shape[0]-shape[0])//(2)+l[i][0]*bin:(tot_shape[0]-shape[0])//(2)+l[i][0]*bin + bin, (tot_shape[1]-shape[1])//(2)+l[i][1]*bin:(tot_shape[1]-shape[1])//(2)+l[i][1]*bin + bin] = (np.pi + y[(tot_shape[0]-shape[0])//(2)+l[i][0]*bin:(tot_shape[0]-shape[0])//(2)+l[i][0]*bin + bin, (tot_shape[1]-shape[1])//(2)+l[i][1]*bin:(tot_shape[1]-shape[1])//(2) + l[i][1]*bin + bin])%(2*np.pi)

      y[(tot_shape[0]-shape[0])//(2)+ each_IsingShape[0] + l[i][0]*bin:(tot_shape[0]-shape[0])//(2)+  each_IsingShape[0] + l[i][0]*bin + bin, (tot_shape[1]-shape[1])//(2)+  each_IsingShape[1] + l[i][1]*bin:(tot_shape[1]-shape[1])//(2)+ each_IsingShape[1] + l[i][1]*bin + bin] = (np.pi + y[(tot_shape[0]-shape[0])//(2)+ each_IsingShape[0] + l[i][0]*bin:(tot_shape[0]-shape[0])//(2)+ each_IsingShape[0] + l[i][0]*bin + bin, (tot_shape[1]-shape[1])//(2)+ each_IsingShape[1] + l[i][1]*bin:(tot_shape[1]-shape[1])//(2) + each_IsingShape[1] +  l[i][1]*bin + bin])%(2*np.pi)
    return y


#each_IsingShape (256,512 )
def flip_np(spin_arr, x, tot_shape, shape, bin, d, each_IsingShape):
    y = copy.copy(x)
    d = 1
    l = [(0,0)]#random.sample(spin_arr,d)
    

    for i in range(d):
      print()
      print("Ising Up: ", np.unique(y[(tot_shape[0]-shape[0])//(2)+l[i][0]*bin:(tot_shape[0]-shape[0])//(2)+l[i][0]*bin + bin, (tot_shape[1]-shape[1])//(2)+l[i][1]*bin:(tot_shape[1]-shape[1])//(2)+l[i][1]*bin + bin]))
      print()
      a = (y[(tot_shape[0]-shape[0])//(2)+ each_IsingShape[0] + l[i][0]*bin:(tot_shape[0]-shape[0])//(2)+  each_IsingShape[0] + l[i][0]*bin + bin, (tot_shape[1]-shape[1])//(2)+  each_IsingShape[1] + l[i][1]*bin:(tot_shape[1]-shape[1])//(2)+ each_IsingShape[1] + l[i][1]*bin + bin])
      print("Ising DOwn: ", np.unique(a))
      if np.unique(a).size == 0:
        print("why empty array: ",l)
      
      # as both Ising are stacked vertically so the y values of Ising spins(i.e. not slm spins) is same so REMOVE " each_IsingShape[1]"  from (tot_shape[1]-shape[1])//(2) +  each_IsingShape[1] + l[i][1]*bin
      # but the vertical axis or x axis is starting pointer for 2 Ising are different so  we ADD "each_IsingShape[0]" to (tot_shape[0]-shape[0])//(2)+ each_IsingShape[0]
      # --------------------------------------> x axis (same pointer to corresponding spins in 2 Ising)
      # |
      # |
      # |
      # |
      # V y axis  Also shape[0] --- > 2*shape[0]
      #start index for Ising 1 = y[256:320, 256:320]
      y[(tot_shape[0]-2*shape[0])//(2)+l[i][0]*bin:(tot_shape[0]-2*shape[0])//(2)+l[i][0]*bin + bin, (tot_shape[1]-shape[1])//(2)+l[i][1]*bin:(tot_shape[1]-shape[1])//(2)+l[i][1]*bin + bin] = (np.pi + y[(tot_shape[0]-shape[0])//(2)+l[i][0]*bin:(tot_shape[0]-shape[0])//(2)+l[i][0]*bin + bin, (tot_shape[1]-shape[1])//(2)+l[i][1]*bin:(tot_shape[1]-shape[1])//(2) + l[i][1]*bin + bin])%(2*np.pi)

      #start index for Ising 2 = y[512:576, 256,320]
      y[(tot_shape[0]-2*shape[0])//(2)+ each_IsingShape[0] + l[i][0]*bin:(tot_shape[0]-2*shape[0])//(2)+  each_IsingShape[0] + l[i][0]*bin + bin, (tot_shape[1]-shape[1])//(2) + l[i][1]*bin:(tot_shape[1]-shape[1])//(2) + l[i][1]*bin + bin] = (np.pi + y[(tot_shape[0]-2*shape[0])//(2)+ each_IsingShape[0] + l[i][0]*bin:(tot_shape[0]-2*shape[0])//(2)+ each_IsingShape[0] + l[i][0]*bin + bin, (tot_shape[1]-shape[1])//(2) + l[i][1]*bin:(tot_shape[1]-shape[1])//(2) +  l[i][1]*bin + bin])%(2*np.pi)
      
      
    return y


bins =  64 # 32 # 2**3 # 8 #32#
outer_bins = 2**4 # 16

# check here 
beta = np.arange(1,0,-0.2)
count=0

area = (2**10,2**10) # (1024, 1024)
mask = np.zeros(area)
d = 2**2


'''
    Done Initialize device
'''
slm_area = np.pi*checkerboard(area,outer_bins)/128 # 0 or 3.14

spinflip = []

loss_arr = []
loss_arr.append(255)
fidelity_arr = []

zeta_IsUp = np.loadtxt('zeta_IsUp.csv', delimiter=",") #1*np.ones((32, 64)) #  np.ones((8,16))#
zeta_IsUp[-1,-1] = 1 


eta_IsDown =  np.loadtxt('eta_IsDown.csv', delimiter=",") # 100*np.ones(zeta_IsUp.shape) #
theta_Matrix = np.loadtxt('theta_Matrix.csv', delimiter=",") # -1 * np.ones((zeta_IsUp.shape[0],zeta_IsUp.shape[1])) # np.sign(np.random.normal(0,1,          (zeta_IsUp.shape[0],zeta_IsUp.shape[1]))+1e-7)#  np.loadtxt('thetaA_Matrix.csv', delimiter=",") 

#theta spin requires matrix A. y = Ax

active_area = (bins*eta_IsDown.shape[0],bins*eta_IsDown.shape[1])# (2**9,2**9) # (256, 512)
spin_arr = spin_tuple(active_area, bins)
#print(theta_Matrix,'\n', theta_Matrix==-1)



check = np.ones((bins*eta_IsDown.shape[0],bins*eta_IsDown.shape[1]))
for i in range(check.shape[0]//2):
    for j in range(check.shape[1]//2):
        check[i*2:i*2+2,j*2:j*2+2] = (-1)**(i+j) # variable = (?1)?? e(itetha ) = cos

check_SLM = np.vstack((check, check))
#print(check_SLM.shape, eta_IsDown.shape)
         
# Generate Upper Ising Model
lat_IsUp = np.zeros((bins*zeta_IsUp.shape[0],bins*zeta_IsUp.shape[1]))

zeta_Max = np.max(zeta_IsUp)
zeta_IsUp = np.arccos(zeta_IsUp/zeta_Max)
for i in range(zeta_IsUp.shape[0]):
    for j in range(zeta_IsUp.shape[1]):
        lat_IsUp[i*bins:i*bins+bins, j*bins:j*bins+bins] = zeta_IsUp[i,j]

# Generate Lower Ising Model
lat_IsDown = np.zeros((bins*eta_IsDown.shape[0],bins*eta_IsDown.shape[1]))

eta_Max = np.max(eta_IsDown)
eta_IsDown = np.arccos(eta_IsDown/eta_Max)
for i in range(eta_IsDown.shape[0]):
    for j in range(eta_IsDown.shape[1]):
        lat_IsDown[i*bins:i*bins+bins, j*bins:j*bins+bins] = eta_IsDown[i,j]


# Generate a mask variable = (?1)?? cos?1 ???? with both Upper and Lower Ising concanted

lat_SLM = np.vstack((lat_IsUp, lat_IsDown))
mask_inner = check_SLM * lat_SLM # mask variable = (?1)?? cos?1 ????
#print(mask_inner.shape,  lat_IsUp.shape, lat_SLM.shape)


# Generate Random spins [Initialization]
spin_ValUp = np.sign(np.random.normal(0,1, 
          (zeta_IsUp.shape[0],zeta_IsUp.shape[1]))+1e-7)

spin_ValDown = copy.deepcopy(spin_ValUp)

spin_ValUp = (spin_ValUp+1)*np.pi/2  
#print(spin_ValUp)
spin_ValDown = (spin_ValDown+1)*np.pi/2 + (np.pi/2) # adding 3pi/2 for v[i] = 1 or pi/2 for v[i] = -1

# @R no copy problem
print(spin_ValDown)
#Multiply y = A.x

spin_ValDown[theta_Matrix == -1] = (np.pi + spin_ValDown[theta_Matrix == -1])%(2*np.pi) # @R


spin_IsUp = np.zeros((bins*zeta_IsUp.shape[0],bins*zeta_IsUp.shape[1]))

for i in range(zeta_IsUp.shape[0]):
    for j in range(zeta_IsUp.shape[1]):
        spin_IsUp[i*bins:i*bins+bins, j*bins:j*bins+bins] = spin_ValUp[i,j]

#print(spin_ValUp.shape, spin_IsUp.shape)
#assert((spin_IsUp.shape == (np.array([bins,bins])*eta_IsDown.shape) ).all() )


spin_IsDown = np.zeros((bins*eta_IsDown.shape[0],bins*eta_IsDown.shape[1]))

for i in range(eta_IsDown.shape[0]):
    for j in range(eta_IsDown.shape[1]):
        spin_IsDown[i*bins:i*bins+bins, j*bins:j*bins+bins] = spin_ValDown[i,j]

#assert((spin_IsDown.shape == (np.array([bins,bins])*eta_IsDown.shape)).all())


spin_SLM = np.vstack((spin_IsUp, spin_IsDown))
assert(spin_SLM.shape == mask_inner.shape )

print(active_area)
mask[(area[0]-2*active_area[0])//2:(area[0]+2*active_area[0])//2, (area[1]-active_area[1])//2:(area[1]+active_area[1])//2] = mask_inner

slm_area[(area[0]-2*active_area[0])//2:(area[0]+2*active_area[0])//2, (area[1]-active_area[1])//2:(area[1]+active_area[1])//2] = spin_SLM



plt.imshow(slm_area)
plt.show()
'''
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
exp_time = 45

###############################################
## Camera initialize and finding exposure time
###############################################
img = []
N = 500

for i in range(N):
    camera.ExposureTime.SetValue(exp_time)  # microsecond
    camera.StartGrabbingMax(1)

    while camera.IsGrabbing():
    # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = camera.RetrieveResult(15000, pylon.TimeoutHandling_ThrowException)

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
             # Access the image data.
            #img = rebin(grabResult.Array, imgbin)
            img.append(np.mean((grabResult.Array)[imgy1:imgy2, imgx1:imgx2])) # [127:129,127:129]))

        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        grabResult.Release()
    exp_time += 5
print(img, "what -->",np.where(np.asarray(img)>=250))
EXP_TIME = 45+5*np.where(np.asarray(img) >= np.max(np.asarray(img)))[0][0] # @R np.where(np.asarray(img)>=250)[0][0]
print("Optimal exposure time is: ", EXP_TIME, " microseconds")
camera.ExposureTime.SetValue(EXP_TIME*1.0) 

print("Set the exposure time ==> %%%%%%%%%%%%%%%%%%%% ", EXP_TIME)

###############################################
## Camera initialize and finding exposure time
###############################################
'''


###############################################
## Start Metropolis Hasting
###############################################
each_IsingSize = (bins*eta_IsDown.shape[0],bins*eta_IsDown.shape[1])
while not (count == len(beta) - 1):
    print("%%% ==> %%^%")
    slm_area2 = flip_np(spin_arr, slm_area, area, active_area, bins, d, each_IsingSize)
    
    #print(printInfo)
    #error = slm.showPhasevalues(slm_area2+mask)
    '''
    thread1 = threading.Thread(target=threading_test)
    thread1.start()
    thread1.join()
    '''
    time.sleep(0.15) 
    '''
    camera.StartGrabbingMax(1)

    while camera.IsGrabbing():
        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            # Access the image data.
            #img = rebin(grabResult.Array, imgbin)
            img = (grabResult.Array)[imgy1:imgy2, imgx1:imgx2]

        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        grabResult.Release()
    
    '''
    #C1 = np.sum(img)/np.size(img)

    #delE = C1 - loss_arr[-1]
    #print(C1, COST[-1], delE)

    p = np.exp(-1*beta[count]*7.0)                


    if 7 <= 0:
        slm_area=slm_area2
        #print("Accepted")
        #loss_arr.append(C1)
        spinflip.append(d)
    elif np.random.choice(a=[0,1], p=[p, 1-p])==0:
        slm_area=slm_area2
        #loss_arr.append(C1)
        spinflip.append(d)
    else:
        slm_area1=slm_area2
        loss_arr.append(loss_arr[-1])
        spinflip.append(0)
        
    count += 1
    final_screen = slm_area2[(area[0]-active_area[0])//2:(area[0]+active_area[0])//2, (area[1]-active_area[1])//2:(area[1]+active_area[1])//2]


    #s1 = np.where(final_screen < np.pi/2 )
    #s2 = np.where(final_screen >= np.pi/2)
    s1 = final_screen < np.pi/2 
    s2 = final_screen >= np.pi/2
    #print(final_screen)
    #print(s1)
    #print(s2)
    #fidelity = (np.sum(number_part[s1])-np.sum(number_part[s2]))/(np.sum(number_part[s1])+np.sum(number_part[s2]))
    #fidelity_arr.append(fidelity)
    
    #print(s1, type(s1))
    magnetization1 = sum(s1)
    print(sum(magnetization1)/(s1.shape[0]*s1.shape[1]) )
    #magnetization2 = sum(s2)
    #print( sum(magnetization2)/(s2.shape[0]*s2.shape[1])
#print(loss_arr)
print(np.mean(final_screen < np.pi/2))
# CLOSING DOWN THE INSTRUMENTS

np.savetxt('final_answer.csv', final_screen, delimiter=",")


# SHOWING ALL THE PLOTS GENERATED
#plt.show()
print('fidelity', fidelity_arr)
print('loss', loss_arr)
