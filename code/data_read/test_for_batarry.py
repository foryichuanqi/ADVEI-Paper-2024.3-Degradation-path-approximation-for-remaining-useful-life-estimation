# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 20:18:57 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:25:26 2023

@author: Administrator
"""

import numpy as np

import scipy as sp

import math

from numpy import matmul as mm
from math import sqrt,pi,log, exp

import xlrd

import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import minimize, rosen, rosen_der

from scipy.stats import linregress

from scipy.stats import norm


import scipy.io as scio

# path = r'F:\桌面11.17\project\RUL_with_highly_small_sample\dataset\IGBTAgingData_04022009\Data\Thermal Overstress Aging with Square Signal at gate and SMU data\Aging Data\Device 2\Device2  1.mat'
# matdata = scio.loadmat(path)
# print(matdata["measurement"][0][0][1][0][0][2][0][0][4])
# x=matdata["measurement"][0][0][1][0][0][2][0][0][4][0]

# plt.plot(range(0,len(x),10000),x[range(0,len(x),10000)])
# plt.show()



invaribale_index=3    #########dtype=[('dt', 'O'), ('gateSignalVoltage', 'O'), ('gateEmitterVoltage', 'O'), ('collectorEmitterVoltage', 'O'), ('collectorEmitterCurrentSignal', 'O')]),





###########feng tian


path = r'F:\桌面11.17\project\RUL_guassion\handled_dataset\fengtian\training.mat'

matdata = scio.loadmat(path)

x=matdata["training"]
plt.plot(range(len(x)),x)
plt.show()


path = r'F:\桌面11.17\project\RUL_guassion\handled_dataset\fengtian\test.mat'

matdata = scio.loadmat(path)

x=matdata["test"]
plt.plot(range(len(x)),x)
plt.show()















#########  nasa





path = r'F:\桌面11.17\project\RUL_guassion\handled_dataset\nasa\B0005_2.mat'

matdata = scio.loadmat(path)

x=matdata["B0005_2"][0]
plt.plot(range(len(x)),x)
# plt.show()

print(x)


path = r'F:\桌面11.17\project\RUL_guassion\handled_dataset\nasa\B0006_2.mat'

matdata = scio.loadmat(path)

x=matdata["B0006_2"][0]
plt.plot(range(len(x)),x)
# plt.show()

print(x)


path = r'F:\桌面11.17\project\RUL_guassion\handled_dataset\nasa\B0007_2.mat'

matdata = scio.loadmat(path)

x=matdata["B0007_2"][0]
plt.plot(range(len(x)),x)
# plt.show()
print(x)



path = r'F:\桌面11.17\project\RUL_guassion\handled_dataset\nasa\B0018_2.mat'

matdata = scio.loadmat(path)

x=matdata["B0018_2"][0]
plt.plot(range(len(x)),x)
plt.show()




# # print(matdata["measurement"][0][0][1][0][0][2][0][0][4])
# x=matdata["measurement"][0][0][1][0][0][2][0][0][invaribale_index][0]

# x=matdata["measurement"][0][0][1][0][0][2][0][0][invaribale_index][0]

# Vces_list=[]

# for i in range(len(matdata["measurement"][0][0][1][0])):
    
#     # Vces_list.append(matdata["measurement"][0][0][1][0][i][2][0][0][invaribale_index][0][110000])
#     Vces_list.append(matdata["measurement"][0][0][1][0][i][2][0][0][invaribale_index][0][-1])
#     # Vces_list.append(matdata["measurement"][0][0][1][0][i][2][0][0][invaribale_index][0][50000])
    
    
# # print(x)

# plt.plot(range(len(Vces_list)),Vces_list)
# plt.show()





# path = r'F:\桌面11.17\project\RUL_with_highly_small_sample\dataset\IGBTAgingData_04022009\Data\Thermal Overstress Aging with Square Signal at gate and SMU data\Aging Data\Device 3\Device3  1.mat'
# matdata = scio.loadmat(path)
# # print(matdata["measurement"][0][0][1][0][0][2][0][0][4])
# x=matdata["measurement"][0][0][1][0][0][2][0][0][invaribale_index][0]

# x=matdata["measurement"][0][0][1][0][0][2][0][0][invaribale_index][0]

# Vces_list=[]

# for i in range(len(matdata["measurement"][0][0][1][0])):
    
#     # Vces_list.append(matdata["measurement"][0][0][1][0][i][2][0][0][invaribale_index][0][110000])    
#     Vces_list.append(matdata["measurement"][0][0][1][0][i][2][0][0][invaribale_index][0][-1])
#     # Vces_list.append(matdata["measurement"][0][0][1][0][i][2][0][0][invaribale_index][0][50000])
    
    
# # print(x)

# plt.plot(range(len(Vces_list)),Vces_list)
# plt.show()




# path = r'F:\桌面11.17\project\RUL_with_highly_small_sample\dataset\IGBTAgingData_04022009\Data\Thermal Overstress Aging with Square Signal at gate and SMU data\Aging Data\Device 4\Device4  1.mat'
# matdata = scio.loadmat(path)
# # print(matdata["measurement"][0][0][1][0][0][2][0][0][4])
# x=matdata["measurement"][0][0][1][0][0][2][0][0][invaribale_index][0]

# x=matdata["measurement"][0][0][1][0][0][2][0][0][invaribale_index][0]

# Vces_list=[]

# for i in range(len(matdata["measurement"][0][0][1][0])):
#     # Vces_list.append(matdata["measurement"][0][0][1][0][i][2][0][0][invaribale_index][0][110000])    
#     Vces_list.append(matdata["measurement"][0][0][1][0][i][2][0][0][invaribale_index][0][-1])
#     # Vces_list.append(matdata["measurement"][0][0][1][0][i][2][0][0][invaribale_index][0][50000])
    
    
# # print(x)

# plt.plot(range(len(Vces_list)),Vces_list)
# plt.show()





# path = r'F:\桌面11.17\project\RUL_with_highly_small_sample\dataset\IGBTAgingData_04022009\Data\Thermal Overstress Aging with Square Signal at gate and SMU data\Aging Data\Device 5\Device5  1.mat'
# matdata = scio.loadmat(path)
# # print(matdata["measurement"][0][0][1][0][0][2][0][0][4])
# x=matdata["measurement"][0][0][1][0][0][2][0][0][invaribale_index][0]

# x=matdata["measurement"][0][0][1][0][0][2][0][0][invaribale_index][0]

# Vces_list=[]

# for i in range(len(matdata["measurement"][0][0][1][0])):

#     # Vces_list.append(matdata["measurement"][0][0][1][0][i][2][0][0][invaribale_index][0][110000])    
#     Vces_list.append(matdata["measurement"][0][0][1][0][i][2][0][0][invaribale_index][0][-1])
#     # Vces_list.append(matdata["measurement"][0][0][1][0][i][2][0][0][invaribale_index][0][50000])
    
    
# # print(x)

# plt.plot(range(len(Vces_list)),Vces_list)
# plt.show()





# epoch_index=6


# path = r'F:\桌面11.17\project\RUL_with_highly_small_sample\dataset\IGBTAgingData_04022009\Data\Thermal Overstress Aging with Square Signal at gate and SMU data\Aging Data\Device 2\Device2  1.mat'
# matdata = scio.loadmat(path)
# # print(matdata["measurement"][0][0][1][0][0][2][0][0][4])
# # x=matdata["measurement"][0][0][1][0][0][2][0][0][invaribale_index][0]

# x=matdata["measurement"][0][0][1][0][epoch_index][2][0][0][invaribale_index][0]

# print(x)
# plt.plot(range(0,len(x),10000),x[range(0,len(x),10000)])
# plt.show()

