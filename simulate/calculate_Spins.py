import numpy as np


zeta_IsUp = np.loadtxt('../zeta_IsUp.csv', delimiter=",") # np.ones((8,16))#np.ones((32, 64)) # 
#zeta_IsUp[-1,-1] = 1                                 


eta_IsDown = np.loadtxt('../eta_IsDown.csv', delimiter=",") # np.ones(zeta_IsUp.shape) # 
theta_Matrix = np.loadtxt('../theta_Matrix.csv', delimiter=",") # -1 * np.ones((zeta_IsUp.shape[0],zeta_IsUp.shape[1])) # np.sign(np.random.normal(0,1,          (zeta_IsUp.shape[0],zeta_IsUp.shape[1]))+1e-7)#  theta spin requires matrix A. y = Ax

'''
Generate the J matrix Coupling constants
'''

Couple_zeta = zeta_IsUp.reshape(1, -1) # (-1, 1 )
Couple_eta = eta_IsDown.reshape(1, -1) # (-1, 1 )
couple_coupling_sign = theta_Matrix.reshape(1, -1)

#print(Couple_zeta.shape, zeta_IsUp.shape,zeta_IsUp.shape[0]*zeta_IsUp.shape[1], np.transpose(Couple_zeta).shape )
Coupling_constants = Couple_zeta*np.transpose(Couple_zeta)
Coupling_constants1 = Couple_eta*np.transpose(Couple_eta)
couple_coupling_sign1 = couple_coupling_sign*np.transpose(couple_coupling_sign)

Coupling_constants = Coupling_constants + couple_coupling_sign1 * Coupling_constants1

Coupling_constants = Coupling_constants# - np.diag(np.diag(Coupling_constants))

print( "Coupling_constants: ", Coupling_constants)

np.savetxt('Coupling_constants.csv', Coupling_constants, delimiter=",")

'''
Digital Annealer generate  J_matrix
''' 
QSIM_coupling = np.loadtxt('Coupling_constants.csv', delimiter=",")
J_Matrix = []
for i in range(QSIM_coupling.shape[0]):
  for j in range(QSIM_coupling.shape[1]):
    if QSIM_coupling[i,j] != 0:
      J_Matrix.append((i, j, QSIM_coupling[i,j]))

assert(QSIM_coupling.shape[0] == QSIM_coupling.shape[1])      
fptr = open("J_Matrix_4x8.txt", "w")
nodes = (QSIM_coupling.shape[0])  
fptr.write(str(nodes) + ' ' + str(len(J_Matrix))+ '\n')

for i  in J_Matrix:
  fptr.write(str(i[0]+1) + ' ' + str(i[1]+1) + ' '+ str(i[2])+ '\n')

fptr.close()

fptr1 = open("linear_4x8.txt", "w")

for i in range(QSIM_coupling.shape[0]):
  fptr1.write(str(0.0) +' ')
fptr1.close()

'''
Check final spins
'''

bins = 64

my_data = np.genfromtxt('final_answer.csv', delimiter=',')
print("Hello world: ", my_data.shape[0]//bins, my_data.shape[1]//bins)
spin_values = np.zeros((my_data.shape[0]//bins, my_data.shape[1]//bins))

print(my_data, my_data.shape)
for i in range(my_data.shape[0]//bins):
  for j in range(my_data.shape[1]//bins):
    spin_values[i, j] = np.sum(my_data[i*bins:i*bins+bins, j*bins:j*bins+bins])

spin_values = spin_values/(bins*bins)

maxcut_spin = np.array(spin_values < np.pi/2, dtype = np.float32)
maxcut_spin[maxcut_spin==0] = -1

print(maxcut_spin)
'''
Maxcut
'''
maxcut_spin1 = np.loadtxt('./DA_spins.txt', delimiter="\t") # DA spins activate
print("(maxcut_spin1 - maxcut_spin).mean(): ",(maxcut_spin1 - maxcut_spin.reshape(-1, 1)).mean())
maxcut_spin = maxcut_spin.reshape(-1, 1)
maxcut = 0.0
for i in range(maxcut_spin.shape[0]):
  for j in range(maxcut_spin.shape[0]):
    maxcut += 0.5*Coupling_constants[i][j]*(1.0 - maxcut_spin[i][0]*maxcut_spin[j][0]) 

print(maxcut_spin, maxcut_spin.shape)

print("maxcut ", maxcut)
'''
Calculate Energy 
'''
print(np.sum(maxcut_spin)/maxcut_spin.shape[0])


totalEnergy = 0.0
for i in range(maxcut_spin.shape[0]):
  perSpinEn = 0.0
  for j in range(maxcut_spin.shape[0]):
    perSpinEn += -1.0 * Coupling_constants[i][j]*(maxcut_spin[j][0]) 
  perSpinEn *= maxcut_spin[i][0]
  totalEnergy += 0.5*perSpinEn

print(totalEnergy)