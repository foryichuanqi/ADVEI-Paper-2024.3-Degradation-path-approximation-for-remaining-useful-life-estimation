# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:46:05 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 22:07:49 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 20:41:41 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 13:04:56 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:49:17 2022

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:26:02 2022

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:28:22 2022

@author: Administrator
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 18:17:31 2022

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:50:53 2022

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:55:56 2022

@author: Administrator

"""



import xlrd

import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import minimize, rosen, rosen_der

from scipy.stats import linregress


shed=-0.45

import numpy as np
import pandas as pd
import os
import pickle
import scipy as sp
import datetime


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




print(os.path.abspath(os.path.join(os.getcwd(), "../..")))
last_last_path=os.path.abspath(os.path.join(os.getcwd(), "../.."))

print(os.path.abspath(os.path.join(os.getcwd(), "..")))
last_path=os.path.abspath(os.path.join(os.getcwd(), ".."))

print(os.path.abspath(os.path.join(os.getcwd(), "../../..")))
last_last_last_path=os.path.abspath(os.path.join(os.getcwd(), "../../.."))


print(os.path.abspath(os.path.join(os.getcwd(), "../../../..")))
last_last_last_last_path=os.path.abspath(os.path.join(os.getcwd(), "../../../.."))





def get_data_list():

    



    path = last_last_last_last_path+r'\dataset\nasa\B0005_2.mat'
    
    matdata = scio.loadmat(path)
    
    CS2_35=matdata["B0005_2"][0]
    # plt.plot(range(len(x)),x)
    # plt.show()
    
    # print(x)
    
    
    path = last_last_last_last_path+r'\dataset\nasa\B0006_2.mat'
    
    matdata = scio.loadmat(path)
    
    CS2_36=matdata["B0006_2"][0]
    # plt.plot(range(len(x)),x)
    # plt.show()
    
    # print(x)
    
    
    path = last_last_last_last_path+r'\dataset\nasa\B0007_2.mat'
    
    matdata = scio.loadmat(path)
    
    CS2_37=matdata["B0007_2"][0]
    # plt.plot(range(len(x)),x)
    # plt.show()
    # print(x)
    
    
    
    path = last_last_last_last_path+r'\dataset\nasa\B0018_2.mat'
    
    matdata = scio.loadmat(path)
    
    CS2_38=matdata["B0018_2"][0]
    # plt.plot(range(len(x)),x)
    # plt.show()

        
        



    fig, ax = plt.subplots()
    # 在生成的坐标系下画折线图
    ax.plot(CS2_35, linewidth=1,c='b',label="B0005_2")
    ax.plot(CS2_36, linewidth=1,c='g',label="B0006_2")
    ax.plot(CS2_37, linewidth=1,c='y',label="B0007_2")
    ax.plot(CS2_38, linewidth=1,c='r',label="B0018_2")

    # 显示图形
    font1 = { 
    'weight' : 'normal',
    'size' : 14,
    }
        
          
        #设置横纵坐标的名称以及对应字体格式
    font2 = {#'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 30,
    }
        
    plt.xlabel('Cycle',font1) #X轴标签
    plt.ylabel("Capacity (Ah)",font1) #Y轴标签
    plt.legend()
    plt.savefig(last_last_last_last_path+r'\figure\by_code\Dataset_nasa_curves_comparision.eps',dpi=800,format='eps',bbox_inches = 'tight')
    plt.savefig(last_last_last_last_path+r'\figure\by_code\Dataset_nasa_curves_comparision.png',dpi=800,format='png',bbox_inches = 'tight')
    plt.show()
    
    return CS2_35,CS2_36,CS2_37,CS2_38
CS2_35,CS2_36,CS2_37,CS2_38=get_data_list()


import scipy.io as sio
mat_array=np.zeros((max(len(CS2_36),len(CS2_36),len(CS2_37),len(CS2_37)),4))
CS2_35=list(np.array(CS2_35)-CS2_35[0])
CS2_36=list(np.array(CS2_36)-CS2_36[0])
CS2_37=list(np.array(CS2_37)-CS2_37[0])
CS2_38=list(np.array(CS2_38)-CS2_38[0])
mat_array[:len(CS2_36),0]=CS2_36
mat_array[:len(CS2_36),1]=CS2_36
mat_array[:len(CS2_37),2]=CS2_37
mat_array[:len(CS2_37),3]=CS2_37
mat_array=-mat_array
count=[len(CS2_36),len(CS2_36),len(CS2_37),len(CS2_37)]
sio.savemat('nasaData03061018.mat', {'KFY': mat_array,'count': count})


print(CS2_35[0])
print(CS2_36[0])
print(CS2_37[0])
print(CS2_38[0])


# 1.8564874208181574
# 2.035337591005598
# 1.89105229539079
# 1.8550045207910817
print(CS2_35)

CS2_35=list(np.array(CS2_35)-CS2_35[0])
CS2_36=list(np.array(CS2_36)-CS2_36[0])
CS2_37=list(np.array(CS2_37)-CS2_37[0])
CS2_38=list(np.array(CS2_38)-CS2_38[0])

print(CS2_35)


print(CS2_35[0])
print(CS2_36[0])
print(CS2_37[0])
print(CS2_38[0])

def get_health_list(CS2_35,shed):
    for i in range(len(CS2_35)):
        if CS2_35[i]<shed:                           ########################################   小于门槛值
            CS235=CS2_35[0:i]
            return CS235




def get_input_out(CS235,CS237,CS236):
    min_len=min(len(CS235),len(CS236),len(CS237))
    
    input_list=[]
    output_list=[]
    
    true_out_list=[]
    
    CS235_health=get_health_list(CS235,shed) 
    CS237_health=get_health_list(CS237,shed) 
    CS236_health=get_health_list(CS236,shed) 
    # CS238=get_health_list(CS2_38,shed)  
    # print(len(CS236_health))   
    
    # print(min(len(CS235),len(CS236),len(CS237)))
    
    # print("ccccccccccccccccccccccccc")
    
    
    
    for i in range(2,len(CS236_health)):

        
        
        
        # one_input.append(np.array([len(CS235_health),len(CS237_health)]))
        
        








        y = [CS235[0], CS235[min_len-1]]
        x = [0, (min_len-1)*(min_len-1)]
        slope_0, intercept_0, r_value, p_value, std_err = linregress(x, y)

        
        y = [CS237[0], CS237[min_len-1]]
        x = [0,(min_len-1)*(min_len-1)]
        slope_1, intercept_1, r_value, p_value, std_err = linregress(x, y)
        
        
        
        
        
        

        one_input=[]
        one_input.append(np.array(CS235[:i]))
        one_input.append(np.array(CS237[:i]))
        
        one_input.append("CS238")
    
        min_len_train=min(len(CS235),len(CS237))
            
        one_input.append(np.array(CS235[:min_len_train]))
            
      
        one_input.append(np.array(CS237[:min_len_train]))
        
        one_input.append(np.array(CS236[:i]))
        
        
        
        
        
        input_list.append(one_input)
        
        one_out=[]
        one_out.append(np.array(CS236[:i]))
        
        output_list.append(one_out)
        
        true_out_list.append(max(len(CS236_health)-i-1,0))





    
    true_out_array=np.array(true_out_list)
    return input_list[:len(CS236_health)],output_list[:len(CS236_health)],true_out_array[:len(CS236_health)]






import random

pred_list=[]
error_list=[]

for xxx in range(1):

    for i in range(1):
        
        
        if i==0:
            
            input_list,output_list,true_out_array=get_input_out(CS2_36,CS2_36,CS2_37)######第三个变量是测试集 ，前两个是训练集
            
      
        
        rul_pred_list=[]
        
        regress_error_list=[]
        
        for j in range(len(input_list)):
            
            # print(j)
            # print("jjjjjjjjjjjjjjjjjjjjjjjjjj")
            index=j
            def objective(x):
                
                integrate_array=input_list[index][0]*x[0]+input_list[index][1]*x[1]+x[2]
                
                error = np.sum(np.square(integrate_array - output_list[index][0]))/len(integrate_array)
                
                return error    
            
            # def objective(x):   #############加入时间权重    np.arange(1,len(integrate_array)+1))
                
            #     integrate_array=input_list[index][0]*x[0]+input_list[index][1]*x[1]
                
            #     error = np.sum(np.dot(np.square(integrate_array - output_list[index][0]),np.arange(1,len(integrate_array)+1)))/len(integrate_array)
                
            #     return error           
            
            
            # x0=np.ones(3)/3
    
            
            x0=np.ones(3)
            x0[2]=0
     
            
            
    
            # sol = minimize(objective, x0 ) 
            sol = minimize(objective, x0 ,method='BFGS', options={'maxiter':(1+1)*200,'gtol':1e-5 }  )     
            u1=sol.x[0] 
            u2=sol.x[1]
            c1=input_list[j][3]
    
            c2=input_list[j][4]
            
            offset_vertical=sol.x[2]
    
    
    
            

            
            

            
            ######补偿误差
            pred_online_cruve=u1*input_list[j][0] +u2*input_list[j][1] +offset_vertical
            # pred_curve=u1*c1+u2*c2  + offset_vertical +      np.mean((input_list[j][5] - pred_online_cruve)[-20:] )      
            pred_curve=u1*c1+u2*c2  + offset_vertical +      np.mean((input_list[j][5] - pred_online_cruve)[int(-min(j*0.2,20)):] )    
    
    #############预测曲线          
            # plt.plot(range(len(pred_curve)), pred_curve)
            # plt.show()
     
    #############
    
            
            ful_pred=len(pred_curve)
            for ii in range(len(pred_curve)):
                
                # if i == len(pred_curve)-1:
                    
                #     ful_pred==i
                
                if pred_curve[ii]<shed:
                    
                    ful_pred=ii
                    
                    break
    
    
            rul_pred_list.append(ful_pred-j-1) #######(-1  because of  begining from 2)
    
    
    
    
    
    #############实时的拟合曲线与实际曲线的所有点的误差        
            # pred_online_cruve=u1*input_list[j][0] +u2*input_list[j][1]                
            # plt.plot(range(len(input_list[j][5])), input_list[j][5] - pred_online_cruve)
            # plt.show()
    
    #################
    
    
                    
        
        rul_pred_array=np.array(rul_pred_list)
        
        error_pred_array=rul_pred_array-true_out_array
        
        error_pred_array=np.maximum(error_pred_array, -error_pred_array)
            # print(sol.x)
            
        # print(error_pred_array.sum())
        # print("xxxxx")
        # print(error_pred_array)
            
            
        fig, ax = plt.subplots()
        # 在生成的坐标系下画折线图
        ax.plot(error_pred_array, linewidth=1)
        
        
        
        # 显示图形
        plt.show()   
         
        
        plt.show()   
         
        pred_list.append(rul_pred_array)
        error_list.append(error_pred_array)
        print(xxx)
print(pred_list)

print(error_list)
        
