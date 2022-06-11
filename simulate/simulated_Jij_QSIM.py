import numpy as np

zeta_IsUp = 0 * np.ones((4,8))#np.ones((32, 64)) # np.loadtxt('Zeta_IsingUp.csv', delimiter=",") # 
zeta_IsUp[-1, -1] = 1

eta_IsDown = -4 * np.ones(zeta_IsUp.shape) # np.loadtxt('Eta_IsingDown.csv', delimiter=",")
theta_Matrix = np.sign(np.random.normal(0,1, 
          (zeta_IsUp.shape[0],zeta_IsUp.shape[1]))+1e-7)#-1 * np.ones((zeta_IsUp.shape[0],zeta_IsUp.shape[1]))

np.savetxt('zeta_IsUp.csv', zeta_IsUp, delimiter=",")
np.savetxt('eta_IsDown.csv', eta_IsDown, delimiter=",")
np.savetxt('theta_Matrix.csv', theta_Matrix, delimiter=",")

print(theta_Matrix)
Couple_zeta = zeta_IsUp.reshape(1, -1) # (-1, 1 )
Couple_eta = eta_IsDown.reshape(1, -1) # (-1, 1 )
couple_coupling_sign = theta_Matrix.reshape(1, -1)

#print(Couple_zeta.shape, zeta_IsUp.shape,zeta_IsUp.shape[0]*zeta_IsUp.shape[1], np.transpose(Couple_zeta).shape )
Coupling_constants = Couple_zeta*np.transpose(Couple_zeta)
Coupling_constants1 = Couple_eta*np.transpose(Couple_eta)
couple_coupling_sign1 = couple_coupling_sign*np.transpose(couple_coupling_sign)

Coupling_constants = Coupling_constants + couple_coupling_sign1 * Coupling_constants1

Coupling_constants = Coupling_constants - np.diag(np.diag(Coupling_constants))
for i in range(Coupling_constants.shape[0]):
  print(Coupling_constants[i,:].sum())
'''
bins = 64

my_data = np.genfromtxt('final_answer.csv', delimiter=',')
print(my_data.shape[0]//bins, my_data.shape[1]//bins)
spin_values = np.zeros((my_data.shape[0]//bins, my_data.shape[1]//bins))

print(my_data, my_data.shape)
for i in range(my_data.shape[0]//bins):
  for j in range(my_data.shape[1]//bins):
    spin_values[i, j] = np.sum(my_data[i*bins:i*bins+bins, j*bins:j*bins+bins])

print(spin_values/(bins*bins))
'''