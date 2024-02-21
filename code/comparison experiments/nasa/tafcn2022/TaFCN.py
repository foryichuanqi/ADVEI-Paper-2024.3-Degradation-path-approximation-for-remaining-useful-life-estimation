# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:37:59 2023

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


# print(os.path.abspath(os.path.join(os.getcwd(), "../../../")))
# last_last_last_path=os.path.abspath(os.path.join(os.getcwd(), "../../.."))
# def get_data_list(CS2_35_cap_dropOutlier):
#     worksheet = xlrd.open_workbook('F:\桌面11.17\project\RUL_guassion\handled_dataset\CACLE\{}.xlsx'.format(CS2_35_cap_dropOutlier))
#     sheet_names= worksheet.sheet_names()
#     print(sheet_names)
#     CS2_35=[]
#     for sheet_name in sheet_names:
#         sheet = worksheet.sheet_by_name(sheet_name)
#         rows = sheet.nrows # 获取行数
#         cols = sheet.ncols # 获取列数，尽管没用到
#         all_content = []
    
    
#         CS2_35 = sheet.col_values(0) # 获取第二列内容， 数据格式为此数据的原有格式（原：字符串，读取：字符串；  原：浮点数， 读取：浮点数）



#     fig, ax = plt.subplots()
#     # 在生成的坐标系下画折线图
#     ax.plot(CS2_35, linewidth=1)
#     # 显示图形
#     plt.show()
    
#     return CS2_35

# CS2_35=get_data_list("CS2_35_cap_dropOutlier")
# CS2_36=get_data_list("CS2_36_cap_dropOutlier")
# CS2_37=get_data_list("CS2_37_cap_dropOutlier")
# CS2_38=get_data_list("CS2_38_cap_dropOutlier")


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
# CS2_35,CS2_36,CS2_37,CS2_38=get_data_list()

CS2_35,CS2_36,CS2_37,CS2_38=get_data_list()


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
        if CS2_35[i]<shed:                           ########################################   小于门槛值
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

    
    # CS235_health=get_health_list(CS235,shed) 
    CS237_health=get_health_list(CS237,shed) 
    CS236_health=get_health_list(CS236,shed) 
    # CS235_health[::-1] 
    
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
import tensorflow as tf
from tfdeterminism import patch
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
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



sequence_length=20







             
# X_train , Y_train , X_test ,  Y_test =get_input_out_3(CS2_35,CS2_36,CS2_37,sequence_length)  
# 


# X_train , Y_train , X_test ,  Y_test =get_input_out_3(CS2_35,CS2_36,CS2_38,sequence_length) 

X_train , Y_train , X_test ,  Y_test =get_input_out_2(CS2_36,CS2_37,sequence_length) 



for FD in [1]: ######['1','2','3','4']
    # if max_life==110 and FD=='1':
    #     continue
    # if max_life==110 and FD=='2':
        # continue
    
    FD_feature_columns=[]
 











    method_name='grid_FD{}_TaFCN_npseed{}_segment_{}'.format(FD,seed,segment)
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
    

    

    
    
    #########np.greater_equal([4, 2, 1], [2, 2, 2])array([ True, True, False])
    #############tf.cast( ) 或者K.cast( ) 是执行 tensorflow 中的张量数据类型转换，比如读入的图片是int8类型的，一定要在训练的时候把图片的数据格式转换为float32.
    
    ################reduce_sum reduce dimensinality and get sum


    

            #return inputs*x   
    

    
    


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


