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

########相比fitting1 多了一个纵轴的自由度
#########采用  新曲线 计算阈值到达时间
 ########################################   小于门槛值    到达寿命

import xlrd

import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import minimize, rosen, rosen_der

from scipy.stats import linregress


shed=-5.5

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

    
    CS2_35=[1.0522460048000002, 2.219000064, 2.2502500608, 2.2502500608, 2.219000064, 2.219000064, 2.1877500671999996, 2.7502500095999998, 2.7502500095999998, 2.7190000128, 3.2189999616, 3.2502499584, 3.2189999616, 3.2502499584, 3.7189999104, 3.6877499136, 3.7189999104, 3.7189999104, 3.7189999104, 3.6877499136, 3.7189999104, 3.6564999167999996, 3.6877499136, 3.6877499136, 4.1877498624, 4.1877498624, 4.2189998592, 4.1877498624, 4.1877498624, 4.1877498624, 4.2189998592, 4.1877498624, 4.2189998592, 4.1877498624, 4.1877498624, 4.1877498624, 4.1877498624, 4.1877498624, 4.6877498112, 4.6564998144, 4.6877498112, 4.7189998079999995, 4.6877498112, 4.6252498176, 4.6564998144, 4.6564998144, 4.6564998144, 4.6564998144, 4.6564998144, 4.6564998144, 4.6252498176, 5.156499763199999, 5.156499763199999, 5.156499763199999, 5.1252497664, 5.0939997696, 5.156499763199999, 5.156499763199999, 5.1252497664, 5.0627497728, 5.0627497728, 4.8439997952, 5.0939997696, 5.1249997824, 5.1562497791999995, 5.2187497728, 5.187499776, 5.2187497728, 5.4062497536, 5.4687497472, 5.374999756799999, 4.6249998336, 5.187499776, 5.1562497791999995, 5.1249997824, 5.2499997696000005, 5.1249997824, 5.187499776, 5.2812497664, 5.2812497664, 5.1562497791999995, 5.2812497664, 5.2187497728, 5.3124997632, 5.2187497728, 5.1562497791999995, 5.2812497664, 5.34374976, 5.2187497728, 5.34374976, 5.2499997696000005, 5.2187497728, 5.4374997504, 5.374999756799999, 5.2812497664, 5.2812497664, 5.374999756799999, 5.34374976, 5.812499711999999, 5.374999756799999, 5.2499997696000005, 5.2812497664, 5.2499997696000005, 5.2499997696000005, 5.2499997696000005, 5.2499997696000005, 5.4062497536, 5.374999756799999, 5.374999756799999, 1.750000128]
    CS2_36=[1.0473631920000002, 2.250000064, 2.2187500672000002, 2.1875000704, 2.250000064, 2.2187500672000002, 2.2812500608, 2.2187500672000002, 2.2187500672000002, 2.2187500672000002, 2.1875000704, 2.1875000704, 3.2187499647999998, 3.187499968, 3.3124999552, 3.187499968, 3.2187499647999998, 3.1562499712, 3.1562499712, 3.1562499712, 3.65624992, 3.65624992, 3.6874999168, 3.6874999168, 3.6249999232, 4.1562498688, 4.124999872, 4.124999872, 4.124999872, 4.5624998272, 4.593749824, 4.593749824, 4.5624998272, 4.593749824, 4.593749824, 4.593749824, 4.624999820799999, 4.593749824, 4.593749824, 5.0937497728, 5.062499776, 4.9999997824, 5.062499776, 5.0937497728, 5.1874997632, 5.1249997696000005, 5.1249997696000005, 5.062499776, 5.062499776, 5.062499776, 5.0312497791999995, 5.062499776, 5.0937497728, 4.7187498112, 5.249999756799999, 5.1562497664, 5.062499776, 5.5624997248, 5.6249997184, 5.3124997504, 5.249999756799999, 5.249999756799999, 5.1562497664, 5.1874997632, 6.1249996672, 5.6999997184, 5.1749997312, 5.4499996928000005, 6.2557499904, 6.2557499904, 6.1932499968, 6.5994999552, 6.3494999808, 6.3182499839999995, 6.4119999744, 6.3807499776, 6.6932499456, 6.2869999872, 6.6307499519999995, 6.4119999744, 6.4432499712, 6.2244999936, 6.2869999872, 6.2244999936, 6.4119999744, 6.3807499776, 6.3494999808, 6.162, 6.162, 6.5994999552, 6.1932499968, 6.1307500032, 6.1307500032, 6.1307500032, 6.1307500032, 6.1307500032, 2.7557503488, 2.7557503488, 2.6932503552]
    CS2_37=[1.0864256944000001, 3.2965001088, 2.79650016, 2.7340001664, 2.6715001728, 2.6090001792, 2.6090001792, 2.6090001792, 2.5465001856000002, 2.5465001856000002, 2.9840001408, 3.6090000768, 3.2965001088, 3.421500096, 3.3590001024, 3.3590001024, 3.3590001024, 3.2965001088, 3.3590001024, 3.421500096, 3.421500096, 3.3590001024, 3.8590000512, 3.9840000384, 3.9215000448, 3.8590000512, 4.359, 4.4214999936, 4.359, 4.2965000064, 4.2965000064, 4.1715000192, 4.2965000064, 4.2965000064, 4.7964999552, 4.7339999616, 4.7964999552, 4.7964999552, 4.7964999552, 4.7964999552, 4.7339999616, 4.7964999552, 4.7964999552, 4.7339999616, 4.7964999552, 4.7964999552, 4.6089999744, 4.7339999616, 4.7964999552, 4.7339999616, 4.7339999616, 4.7339999616, 4.7964999552, 4.7339999616, 4.7339999616, 4.7964999552, 4.6089999744, 4.671499968, 4.671499968, 4.7339999616, 4.6089999744, 4.6089999744, 4.7339999616, 4.7964999552, 4.7339999616, 4.983999936, 4.983999936, 4.7964999552, 4.9214999424, 5.2339999104, 4.983999936, 5.0464999295999995, 4.983999936, 4.9685000576, 4.9377500544, 5.5002499968, 5.7189999744, 5.805000038399999, 6.5237499647999995, 5.7737500416, 6.5237499647999995, 6.5237499647999995, 6.0550000128, 5.4925000704, 5.7425000448, 6.4612499712, 6.7112499456, 6.0550000128, 5.9612500224, 6.0862500096, 6.336249984, 6.2737499904, 6.492499968, 5.9300000256, 5.9925000191999995, 5.9925000191999995, 6.023750015999999, 5.9925000191999995, 5.805000038399999, 6.117500006399999, 6.1487500032, 6.1487500032, 6.1487500032, 6.1487500032, 2.8362503423999996, 2.8050003456, 2.8362503423999996, 3.2737502976]
    CS2_38=[2.4536132784, 2.4609374976, 2.2505000704, 2.1880000768000003, 2.15675008, 2.4380000512000004, 2.2817500672, 2.2505000704, 2.781750016, 2.7505000192, 2.5942500352, 2.7505000192, 2.7505000192, 2.6567500288000003, 2.781750016, 2.7505000192, 2.6880000256, 3.250499968, 3.250499968, 3.250499968, 3.250499968, 3.2817499648000004, 3.250499968, 3.250499968, 3.2192499712, 3.2192499712, 3.250499968, 3.250499968, 3.71924992, 3.71924992, 3.6879999232, 4.2504998656, 4.2192498688, 4.2192498688, 4.2192498688, 4.2192498688, 4.2192498688, 4.2192498688, 4.6879998208, 4.7192498176, 4.7192498176, 4.6879998208, 4.7192498176, 4.7192498176, 4.7192498176, 4.656749824, 4.6879998208, 4.5942498304, 4.7192498176, 4.6879998208, 4.6879998208, 5.2192497664, 5.2192497664, 5.1879997696, 5.0004997888, 5.3129997568, 5.1879997696, 5.250499763200001, 5.0629997824, 5.3129997568, 5.28174976, 5.28174976, 5.250499763200001, 5.3129997568, 5.3129997568, 5.250499763200001, 5.1567497728, 5.3129997568, 5.594249727999999, 5.9379996928, 6.281749657600001, 6.0004996863999995, 5.906749696, 5.906749696, 6.0942496768000005, 5.9692496896, 5.7817497088, 5.906749696, 5.7817497088, 5.875499699200001, 5.656749721600001, 5.125499776, 6.0942496768000005, 5.8129997056, 5.875499699200001, 5.9379996928, 5.875499699200001, 5.906749696, 5.8442497024, 5.9379996928, 5.8129997056, 5.906749696, 5.8442497024, 5.8442497024, 5.906749696, 6.0942496768000005, 5.9379996928, 5.9379996928, 5.9379996928, 5.9379996928, 6.3129996544, 6.281749657600001, 5.9692496896, 2.7505000192, 2.7505000192, 2.7505000192]
    
    
     

        
        



    fig, ax = plt.subplots()
    # 在生成的坐标系下画折线图
    ax.plot(CS2_35, linewidth=1,c='b',label="Device2")
    ax.plot(CS2_36, linewidth=1,c='g',label="Device3")
    ax.plot(CS2_37, linewidth=1,c='y',label="Device4")
    ax.plot(CS2_38, linewidth=1,c='r',label="Device5")

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
    plt.savefig(last_last_last_last_path+r'\figure\by_code\Dataset_IGBT_curves_comparision.eps',dpi=800,format='eps',bbox_inches = 'tight')
    plt.savefig(last_last_last_last_path+r'\figure\by_code\Dataset_IGBT_curves_comparision.png',dpi=800,format='png',bbox_inches = 'tight')
    plt.show()
    
    return CS2_35[:-5],CS2_36[:-5],CS2_37[:-5],CS2_38[:-5]



CS2_35,CS2_36,CS2_37,CS2_38=get_data_list()


print(CS2_35)

CS2_35=list(-(np.array(CS2_35)-CS2_35[0]))
CS2_36=list(-(np.array(CS2_36)-CS2_36[0]))
CS2_37=list(-(np.array(CS2_37)-CS2_37[0]))
CS2_38=list(-(np.array(CS2_38)-CS2_38[0]))

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
  
            pred_curve=u1*c1+u2*c2  + offset_vertical +      np.mean((input_list[j][5] - pred_online_cruve) [int(-min(j*0.2,20)):] )       
     
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
         
        pred_list.append(rul_pred_array)
        error_list.append(error_pred_array)
        print(xxx)
print(pred_list)
print(error_list) 


# 0
# rul_pred_array
# [19, 23, 22, 21, 24, 23, 22, 21, 20, 19, 14, 27, 44, 50, 49, 48, 47, 46, 45, 51, 73, 49, 41, 40, 43, 42, 37, 36, 35, 41, 40, 39, 38, 30, 29, 28, 27, 20, 25, 28, 23, 22, 25, 24, 23, 25, 24, 20, 19, 18, 17, 16, 11, 14, 16, 15, 19, 36, 35, 11, 15, 9, 8, 30, 29, 5, 4, 26, 25, 24, 23, 22, 21, 20]
# true_out_array
# [73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
# error_pred_array
# [54, 49, 49, 49, 45, 45, 45, 45, 45, 45, 49, 35, 17, 10, 10, 10, 10, 10, 10, 3, 20, 3, 10, 10, 6, 6, 10, 10, 10, 3, 3, 3, 3, 10, 10, 10, 10, 16, 10, 6, 10, 10, 6, 6, 6, 3, 3, 6, 6, 6, 6, 6, 10, 6, 3, 3, 2, 20, 20, 3, 2, 3, 3, 20, 20, 3, 3, 20, 20, 20, 20, 20, 20, 20]
# 1
# rul_pred_array
# [19, 23, 22, 21, 24, 23, 22, 21, 20, 19, 14, 27, 44, 50, 49, 48, 47, 46, 45, 51, 73, 49, 41, 40, 43, 42, 37, 36, 35, 41, 40, 39, 38, 30, 29, 28, 27, 20, 25, 28, 23, 22, 25, 24, 23, 25, 24, 20, 19, 18, 17, 16, 11, 14, 16, 15, 19, 36, 35, 11, 15, 9, 8, 30, 29, 5, 4, 26, 25, 24, 23, 22, 21, 20]
# true_out_array
# [73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
# error_pred_array
# [54, 49, 49, 49, 45, 45, 45, 45, 45, 45, 49, 35, 17, 10, 10, 10, 10, 10, 10, 3, 20, 3, 10, 10, 6, 6, 10, 10, 10, 3, 3, 3, 3, 10, 10, 10, 10, 16, 10, 6, 10, 10, 6, 6, 6, 3, 3, 6, 6, 6, 6, 6, 10, 6, 3, 3, 2, 20, 20, 3, 2, 3, 3, 20, 20, 3, 3, 20, 20, 20, 20, 20, 20, 20]