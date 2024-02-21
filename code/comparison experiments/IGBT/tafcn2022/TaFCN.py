# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:37:59 2023

@author: Administrator
"""
########正负 门槛值  大于小于



import xlrd

import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import minimize, rosen, rosen_der

from scipy.stats import linregress




import numpy as np
import pandas as pd
import os
import pickle
import scipy as sp
import datetime



timestep=1

shed=5.5



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
shed=4.5

# print(CS2_35)

CS2_35=list(np.array(CS2_35)-CS2_35[0])
CS2_36=list(np.array(CS2_36)-CS2_36[0])
CS2_37=list(np.array(CS2_37)-CS2_37[0])
CS2_38=list(np.array(CS2_38)-CS2_38[0])

# print(CS2_35)


# print(CS2_35[0])
# print(CS2_36[0])
# print(CS2_37[0])
# print(CS2_38[0])

def get_health_list(CS2_35,shed):
    for i in range(len(CS2_35)):
        if CS2_35[i]>shed:                           ########################################   小于门槛值
            CS235=CS2_35[0:i]
            return list(CS235)

# CS235=get_health_list(CS2_35,shed) 
# CS236=get_health_list(CS2_36,shed) 
# CS237=get_health_list(CS2_37,shed) 
# CS238=get_health_list(CS2_38,shed)   

# fig, ax = plt.subplots()
# # 在生成的坐标系下画折线图
# ax.plot(CS235, linewidth=1)
# ax.plot(CS236, linewidth=1)
# ax.plot(CS237, linewidth=1)
# ax.plot(CS238, linewidth=1)

# # 显示图形
# plt.show()

def get_input_out_2(CS236,CS237,time_windows):

    print(CS237)
    # CS235_health=get_health_list(CS235,shed) 
    CS237_health=get_health_list(CS237,shed) 
    CS236_health=get_health_list(CS236,shed) 
    # CS235_health[::-1] 
    # print(CS236_health)
    
    # print(CS235_health)
    
    # CS235_health=list(reversed(CS235_health))
    # print(CS23_health)
    
    
    
    CS236_health=list(reversed(CS236_health))
    CS237_health=list(reversed(CS237_health))
    
    # for i in range(time_windows-1-2):
    #     CS235_health.append(0)
        
    for i in range(time_windows-1-2):
        CS236_health.append(0)    
    
    for i in range(time_windows-1-2):
        CS237_health.append(0)
        
    # CS235_health=list(reversed(CS235_health))
    CS236_health=list(reversed(CS236_health))
    CS237_health=list(reversed(CS237_health))   
    
    x_train_list=[]
    y_train_list=[]
    # for i in range(len(CS235_health)-time_windows+1):        
    #     x_train_list.append(np.array(CS235_health[i:i+time_windows]))
    #     y_train_list.append(len(CS235_health)-time_windows+1-1-i) 
               
    for i in range(len(CS236_health)-time_windows+1):        
        x_train_list.append(np.array(CS236_health[i:i+time_windows]))  
        y_train_list.append(len(CS236_health)-time_windows+1-1-i)        
    x_train_array=np.array(x_train_list)
    y_train_array=np.array(y_train_list)
    
    
    x_test_list=[]
    y_test_list=[]
    for i in range(len(CS237_health)-time_windows+1):        
        x_test_list.append(np.array(CS237_health[i:i+time_windows]))
        y_test_list.append(len(CS237_health)-time_windows+1-1-i) 
                  
    x_test_array=np.array(x_test_list)
    y_test_array=np.array(y_test_list)   
    
    return x_train_array , y_train_array , x_test_array ,  y_test_array









def get_input_out_3(CS235,CS236,CS237,time_windows):
    # min_len=min(len(CS235),len(CS236),len(CS237))
    
    # input_list=[]
    # output_list=[]
    
    # true_out_list=[]
    
    CS235_health=get_health_list(CS235,shed) 
    CS236_health=get_health_list(CS236,shed) 
    CS237_health=get_health_list(CS237,shed)     
    
    print(len(CS237_health))
    print("hhhhhhhhhhhh")
    # CS235_health[::-1] 
    
    # print(CS235_health)
    
    CS235_health=list(reversed(CS235_health))
    # print(CS23_health)
    
    
    
    CS236_health=list(reversed(CS236_health))
    CS237_health=list(reversed(CS237_health))
    
    for i in range(time_windows-1-2):
        CS235_health.append(0)
        
    for i in range(time_windows-1-2):
        CS236_health.append(0)    
    
    for i in range(time_windows-1-2):
        CS237_health.append(0)
        
    CS235_health=list(reversed(CS235_health))
    CS236_health=list(reversed(CS236_health))
    CS237_health=list(reversed(CS237_health))   
    
    x_train_list=[]
    y_train_list=[]
    for i in range(len(CS235_health)-time_windows+1):        
        x_train_list.append(np.array(CS235_health[i:i+time_windows]))
        y_train_list.append(len(CS235_health)-time_windows+1-1-i) 
               
    for i in range(len(CS236_health)-time_windows+1):        
        x_train_list.append(np.array(CS236_health[i:i+time_windows]))  
        y_train_list.append(len(CS236_health)-time_windows+1-1-i)        
    x_train_array=np.array(x_train_list)
    y_train_array=np.array(y_train_list)
    
    
    x_test_list=[]
    y_test_list=[]
    for i in range(len(CS237_health)-time_windows+1):        
        x_test_list.append(np.array(CS237_health[i:i+time_windows]))
        y_test_list.append(len(CS237_health)-time_windows+1-1-i) 
                  
    x_test_array=np.array(x_test_list)
    y_test_array=np.array(y_test_list)   
    
    print(y_test_array.shape)
    print("jjjjjjjjjj")
    
    return x_train_array , y_train_array , x_test_array ,  y_test_array
    
    
    
    
# x_train_array , y_train_array , x_test_array ,  y_test_array=get_input_out_3(CS2_35,CS2_36,CS2_37,20)      

# x_train_array , y_train_array , x_test_array ,  y_test_array=get_input_out_2(CS2_36,CS2_37,20)   








#import tensorflow as tf
import os
import logging
import numpy as np
#from numpy import trans
import matplotlib.pyplot as plt
#import tensorflow as tf
# import CMAPSSDataset
import pandas as pd
import datetime
import keras
from keras.layers import Lambda
import math
import keras.backend as K
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tfdeterminism import patch
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
# from tf import keras
patch()
# tf.random.set_seed(0)
#import keras
#flags = tf.flags
#flags.DEFINE_string("weights", None, 'weights of the network')################# the file path of weights
#flags.DEFINE_integer("epochs", 100, 'train epochs')
#flags.DEFINE_integer("batch_size", 32, 'batch size for train/test')
#flags.DEFINE_integer("sequence_length", 32, 'sequence length')
#flags.DEFINE_boolean('debug', False, 'debugging mode or not')
#FLAGS = flags.FLAGS

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true),axis=0))##################  axis=0

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())




    
segment=3



run_times=10



nb_epochs=2000           #200
batch_size=64    ## 64        #####300
# sequence_length=31    ############# min31  max303

patience=50
patience_reduce_lr=20





seed=2



num_filter1=64
num_filter2=128
num_filter3=64



kernel1_size=8
kernel2_size=5
kernel3_size=3











             



# X_train , Y_train , X_test ,  Y_test =get_input_out_3(CS2_35,CS2_36,CS2_38,sequence_length) 


sequence_length=20



X_train , Y_train , X_test ,  Y_test =get_input_out_2(CS2_36,CS2_37,sequence_length)


for FD in [1]: ######['1','2','3','4']
    # if max_life==110 and FD=='1':
    #     continue
    # if max_life==110 and FD=='2':
        # continue
    
    FD_feature_columns=[]
 











    method_name='grid_FD{}_TaFCN_IGBT_npseed{}_segment_{}'.format(FD,seed,segment)
    # method_name='FCN_RUL_1out_train_split_test'
    dataset='cmapssd'
    
    
    def unbalanced_penalty_score_1out(Y_test,Y_pred) :
          
        s=0    
        for i in range(len(Y_pred)):
            if Y_pred[i]>Y_test[i]:
                s=s+math.exp((Y_pred[i]-Y_test[i])/10)-1
            else:
                s=s+math.exp((Y_test[i]-Y_pred[i])/13)-1    
        print('unbalanced_penalty_score{}'.format(s))
        return s  
      
    def error_range_1out(Y_test,Y_pred) :           
        error_range=(Y_test-Y_pred).min(),(Y_test-Y_pred).max()
        print('error range{}'.format(error_range))
        return error_range
    
 
    print(X_train.shape)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1,1)
    
    
    X_test=X_test.reshape(X_test.shape[0],X_train.shape[1],1,1)
    
    # x_train_array , y_train_array , x_test_array ,  y_test_array=get_input_out_2(CS2_36,CS2_37,20)   

    
    
    import six
    
    import keras.backend as K
    from keras.utils.generic_utils import deserialize_keras_object
    from keras.utils.generic_utils import serialize_keras_object
    from tensorflow.python.ops import math_ops
    from tensorflow.python.util.tf_export import tf_export
    
    
    
    
    
    from tensorflow.python.ops import math_ops
    

    

    
    


    
    


    # reshape_size=len(FD_feature_columns)*int((sequence_length/3))
    def FCN_model():
    #    in0 = keras.Input(shape=(sequence_length,train_feature_slice.shape[1]))  # shape: (batch_size, 3, 2048)
    #    in0_shaped= keras.layers.Reshape((train_feature_slice.shape[1],sequence_length,1))(in0)   
    
        in0 = keras.Input(shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]),name='layer_13')  # shape: (batch_size, 3, 2048)
    #    begin_senet=SeBlock()(in0)
        x = keras.layers.AveragePooling2D(pool_size=(int(sequence_length/segment), 1), strides=int(sequence_length/segment),name='layer_12')(in0)
        # x = keras.layers.Reshape((-1,1))(x) 
        
        # x = keras.layers.Reshape((len(FD_feature_columns)*int((sequence_length/3)),))(x)             
        x = keras.layers.Reshape((-1,))(x)               
        # x = keras.layers.GlobalAveragePooling2D()(in0)
        x = keras.layers.Dense(1, use_bias=False,activation=keras.activations.relu)(x)
        kernel = keras.layers.Dense(1, use_bias=False,activation=keras.activations.hard_sigmoid,name='layer_11')(x)
        begin_senet= keras.layers.Multiply(name='layer_10')([in0,kernel])    #给通道加权重
     
    
       
    
    #     conv0 = keras.layers.
        
        
        conv0 = keras.layers.Conv2D(num_filter1, kernel1_size, strides=1, padding='same',name='layer_9')(begin_senet)
        conv0 = keras.layers.BatchNormalization()(conv0)
        conv0 = keras.layers.Activation('relu',name='layer_8')(conv0)
        
    #    conv0 = keras.layers.Dropout(dropout)(conv0)
        conv0 = keras.layers.Conv2D(num_filter2, kernel2_size, strides=1, padding='same',name='layer_7')(conv0)
        conv0 = keras.layers.BatchNormalization()(conv0)
        conv0 = keras.layers.Activation('relu',name='layer_6')(conv0)
        
    #    conv0 = keras.layers.Dropout(dropout)(conv0)
        conv0 = keras.layers.Conv2D(num_filter3, kernel3_size, strides=1, padding='same',name='layer_5')(conv0)
        conv0 = keras.layers.BatchNormalization()(conv0)
        conv0 = keras.layers.Activation('relu',name='layer_4')(conv0)
        conv0 = keras.layers.GlobalAveragePooling2D(name='layer_3')(conv0)
        conv0 = keras.layers.Dense(64, activation='relu',name='layer_2')(conv0)
        out = keras.layers.Dense(1, activation='relu',name='layer_1')(conv0)
    
        
        
    
    
    
        model = keras.models.Model(inputs=in0, outputs=[out])    
    
        return model
    
    
    # ##############shuaffle the data
    np.random.seed(seed)
    index=np.arange(X_train.shape[0])
    np.random.shuffle(index,)
    
     
    X_train=X_train[index]#X_train是训练集，y_train是训练标签
    Y_train=Y_train[index]
    
    #X_train, Xtest, Y_train, ytest = train_test_split(X_train, Y_train, test_size=0.7, random_state=0)
    
    
    if __name__ == '__main__':
    
        error_record=[]
        index_record=[]
        unbalanced_penalty_score_record=[]
        error_range_left_record=[]
        error_range_right_record=[]
        index_min_val_loss_record,min_val_loss_record=[],[]
        
        if os.path.exists(r"F:\桌面11.17\project\RUL\experiments_result\method_error_txt\{}.txt".format(method_name)):os.remove(r"F:\桌面11.17\project\RUL\experiments_result\method_error_txt\{}.txt".format(method_name))
    
     
    

    
        rul_pred_array_list=[]
        true_out_array_list=[]          
        error_pred_array_list=[]
                    
    #######             single  output                
    
        for i in range(run_times):
            print('xxx')
 
            model=FCN_model()
            plot_model(model, to_file=r"F:\桌面11.17\project\RUL\Flatten.png", show_shapes=True)#########to_file='Flatten.png',r"F:\桌面11.17\project\RUL\model\FCN_RUL_1out_train_valid_test\{}.h5
            
            optimizer = keras.optimizers.Adam()
            model.compile(loss='mse',#loss=root_mean_squared_error,
                          optimizer=optimizer,
                          metrics=[root_mean_squared_error])
             
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                              patience=patience_reduce_lr, min_lr=0.0001) 
            

    #                  verbose=1, validation_split=VALIDATION_SPLIT, callbacks = [reduce_lr])   
            model_name='{}_dataset_{}_log{}_time{}'.format(method_name,dataset,i,datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
            earlystopping=keras.callbacks.EarlyStopping(monitor='loss',patience=patience,verbose=1)
            modelcheckpoint=keras.callbacks.ModelCheckpoint(monitor='loss',filepath=r"F:\桌面11.17\project\RUL\model\FCN_RUL_1out_train_valid_test\{}.h5".format(model_name),save_best_only=True,verbose=1)
            hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                      verbose=1, validation_data=(X_test, Y_test), callbacks = [reduce_lr,earlystopping,modelcheckpoint])   
    #        hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
    #                  verbose=1, validation_data=(X_test, Y_test), callbacks = [reduce_lr,earlystopping,modelcheckpoint])   
            log = pd.DataFrame(hist.history)
            log.to_excel(r"F:\桌面11.17\project\RUL\experiments_result\log\{}_dataset_{}_log{}_time{}.xlsx".format(method_name,dataset,i,datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
            
            print(hist.history.keys())
            epochs=range(len(hist.history['loss']))
            plt.figure()
            plt.plot(epochs,hist.history['loss'],'b',label='Training loss')
            plt.plot(epochs,hist.history['val_loss'],'r',label='Validation val_loss')
            plt.title('Traing and Validation loss')
            plt.legend()
            plt.show()

    
            
    #        model=keras.models.load_model(r"F:\桌面11.17\project\RUL\model\FCN_RUL_1out_train_valid_test\{}.h5".format(model_name),custom_objects={'root_mean_squared_error': root_mean_squared_error,'Smooth':Smooth,'SeBlock':SeBlock})
            model=keras.models.load_model(r"F:\桌面11.17\project\RUL\model\FCN_RUL_1out_train_valid_test\{}.h5".format(model_name),custom_objects={'root_mean_squared_error': root_mean_squared_error})
            for layer in model.layers:
                layer.trainable=False        
    #        score = model.evaluate(X_test, Y_test)  ############forbid evaluate!!!!!!!!!!!!!!!!!!
    #        print('score[1]:{}'.format(score[1]))    ############forbid evaluate!!!!!!!!!!!!!!!!!!    
            
            Y_pred=model.predict(X_test)
    #        rmse=root_mean_squared_error(Y_test,Y_pred)
    #        with tf.Session() as sess:
    #            print(rmse.eval())
            rmse_value=rmse(Y_test,Y_pred)
            # print('rmse:{}'.format(rmse_value))
            

            rul_pred_array=np.array(Y_pred)
            rul_pred_array=rul_pred_array.reshape(rul_pred_array.shape[0])
            
            # print(rul_pred_array.shape)
            
            true_out_array=np.array(Y_test)
            
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
             
            
            # print(i)
            # print("rul_pred_array") 
            # print(list(rul_pred_array))
            # print("true_out_array") 
            # print(list(true_out_array))
            # print("error_pred_array") 
            # print(list(error_pred_array))  
            
            rul_pred_array_list.append(rul_pred_array)
            true_out_array_list.append(true_out_array)            
            error_pred_array_list.append(error_pred_array)
        rul_pred_array=np.mean(rul_pred_array_list,axis=0)
        true_out_array=np.mean(true_out_array_list,axis=0)
        error_pred_array=np.mean(error_pred_array_list,axis=0)


        print(i)
        print("rul_pred_array") 
        print(list(rul_pred_array))
        print("true_out_array") 
        print(list(true_out_array))
        print("error_pred_array") 
        print(list(error_pred_array))  



