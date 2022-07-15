import deepxde as dde
import numpy as np

import yaml
import pandas as pd
import time
import datetime
import os

#from deepxde.backend import tf
#from deepxde.backend import torch
import torch
from scipy.special import gamma
from scipy.special import lambertw

#import mayavi.mlab as mlab
import matplotlib.pyplot as plt

#from draw_counter import random_sphere_points, uniform_sample_points
from PIL import Image

torch.set_default_tensor_type(torch.cuda.FloatTensor)

## 读取Config文件
with open('./Config/train_20220715.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
config['NetWidth']

## 单位统一至cm，定义各个参数
D = 0.585
Rsd = 17.5573
#Rwall = 2.654
Rwall = 2.56401
rpump = 0.75
q = 5
rcell = 1.5

Rrel = Rsd + q*Rwall
I = config['LightIntensity']*1e-3*1e4 # kg*(cm)^2/s^3
T = config['Temperature']
k1 = 10**(-20)*10**8 # kg*cm^4/s^2
k2 = 10**(-17)*10**4 # cm^2
k3 = 1 # cm^-3
sigm = 2.34601*10**(-13) # cm^2

n = k3*(1/(T+273.15))*10**(21.866+4.402-4453/(T+273.15))

'''
读取Ref文件，COMSOL仿真结果
'''

# 读取z-axis实验结果
filename = './COMSOL_data/dual_light_zaxis.csv'
#df = pd.read_csv(filename, header=9, names=['x', 'y', 'z', 'Pz'])
df = pd.read_csv(filename, header=9, names=['Pz'])
ZAxis_COMSOL = np.asarray(df['Pz'])

# 读取3D仿真结果文件
filename_3D = './COMSOL_data/3D_dual_light.csv'
#df = pd.read_csv(filename, header=9, names=['x', 'y', 'z', 'Pz'])
df_3D = pd.read_csv(filename_3D, header=9, names=['x', 'y', 'z', 'Pz'])
axis_3D_x = np.asarray(df_3D['x'])
axis_3D_y = np.asarray(df_3D['y'])
axis_3D_z = np.asarray(df_3D['z'])
axis_3D_Pz = np.asarray(df_3D['Pz'])
axis_3D_axis = np.stack((axis_3D_x, axis_3D_y, axis_3D_z), axis=1)

## PINNs，计算z-axis的极化率数据. z_fitness 拟合结果

def Z_Axis(sample_num=300, rcell=1.5):
    x = 0
    y = 0
    z_sequence = np.linspace(-rcell, rcell, sample_num)
    sample_data = []
    
    for z in z_sequence:
        sample_data.append((x, y, z))
        
    sample_data = np.asarray(sample_data, dtype=np.float32)
    return sample_data


def fitness_metric(y_pred, y_ref):
    # 优化度拟合矩阵
    y_l2_error = np.linalg.norm(y_pred - y_ref, ord=2) # L2 范数
    y_pred_sum = np.sum(y_pred**2)
    
    fitness = 1 - np.sqrt(y_l2_error / y_pred_sum)
    return fitness

def bloch_pde(x_in, y_in):
    ''' x_in: (x,y,z) axis
        y_in: Pz polarization
    '''
    x_in_copy = x_in.detach().cpu().numpy()
    x = x_in_copy[:, 0:1]
    y = x_in_copy[:, 1:2]
    z = x_in_copy[:, 2:3]
    
#     sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
#     x_arr = x.eval(session=sess)
    dy_xx = dde.grad.hessian(y_in, x_in, i=0, j=0)
    dy_yy = dde.grad.hessian(y_in, x_in, i=1, j=1)
    dy_zz = dde.grad.hessian(y_in, x_in, i=2, j=2) # dy/dxidxj
    
    plog1 = lambertw((sigm*I/(Rrel*k1))*np.exp((sigm*I/(Rrel*k1))-n*sigm*z))
    plog2 = lambertw((sigm*I/(Rrel*k1))*np.exp((sigm*I/(Rrel*k1))+n*sigm*z))
    Rop0_z = k2*(Rrel/sigm)*(np.real(plog1) + np.real(plog2))
    Rop0_z = np.asarray(Rop0_z)    
    
    Rop = Rop0_z * np.exp(-2*(x**2+y**2)/rpump**2)
    Rop = torch.from_numpy(Rop).cuda()
    #Rop = Rop0
    
    return -D*(dy_xx+dy_yy+dy_zz) + ((Rop+Rsd)/q+Rwall)*y_in - Rop/q


def func_boundary(x_in, Pz, numpy_X):
    x_in_copy = x_in.detach().cpu().numpy()
    x = x_in_copy[:, 0:1]
    y = x_in_copy[:, 1:2]
    z = x_in_copy[:, 2:3]
    
    dPz_x = dde.grad.jacobian(Pz, x_in, i=0, j=0) # This is right
    dPz_y = dde.grad.jacobian(Pz, x_in, i=0, j=1)
    dPz_z = dde.grad.jacobian(Pz, x_in, i=0, j=2)
    
    #Rop = Rop0 * np.exp(-2*(x**2+y**2)/rpump**2)
    
    plog1 = lambertw((sigm*I/(Rrel*k1))*np.exp((sigm*I/(Rrel*k1))-n*sigm*z))
    plog2 = lambertw((sigm*I/(Rrel*k1))*np.exp((sigm*I/(Rrel*k1))+n*sigm*z))
    Rop0_z = k2*(Rrel/sigm)*(np.real(plog1) + np.real(plog2))
    Rop0_z = np.asarray(Rop0_z)      
    
    Rop = Rop0_z * np.exp(-2*(x**2+y**2)/rpump**2)
    
    second_term = np.sqrt(1/2*np.abs(Rop*D))
    
    # tranfer back to torch.tensor
    second_term = torch.from_numpy(second_term).cuda()
    Rop = torch.from_numpy(Rop).cuda()
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    z = torch.from_numpy(z).cuda()
    
    return (D/rcell*(x*dPz_x + y*dPz_y + z*dPz_z) + Pz*second_term)


geom = dde.geometry.Sphere([0, 0, 0], rcell)
bc = dde.icbc.OperatorBC(geom, func_boundary, lambda _, on_boundary: on_boundary) # 函数调研下，operatorBC

data = dde.data.PDE(geom, bloch_pde, bc, num_domain=25000, num_boundary=3000)

## 核心训练程序
for depth in config['NetDepth']:
    for width in config['NetWidth']:
        net = dde.nn.FNN([3] + [width] * depth + [1], "tanh", "Glorot normal")
        # net.apply_output_transform(
        #     lambda x, y: (1 - tf.reduce_sum(x ** 2, axis=1, keepdims=True)) * y
        # )

        model = dde.Model(data, net)
        model.compile("adam", lr=1e-4)
        #losshistory, train_state = model.train(epochs=8000, model_save_path='Bloch_Rop_v1')
        losshistory, train_state = model.train(epochs=config['TrainEpoch'])
        print('Train finish : {0} width {1} depth'.format(width, depth))

        YZ_dir_path = './result/Intensity{0}_Temp{1}_NetDepth{2}_Width{3}'.format(config['LightIntensity'], T, depth,\
                                                                                    width)
        if not os.path.exists(YZ_dir_path):
            os.mkdir(YZ_dir_path)
        zAxis_pred = model.predict(Z_Axis())[:, 0]
        zAxis_df = pd.DataFrame(zAxis_pred)
        zAxis_path = os.path.join(YZ_dir_path, 'zAxis_pred.csv')
        zAxis_df.to_csv(zAxis_path)       
        z_fitness = fitness_metric(zAxis_pred, ZAxis_COMSOL)

        ## 计算三维数据预测结果
        y_pred_3D = model.predict(axis_3D_axis)[:, 0]
        y_ref = axis_3D_Pz

        #L2_mean = np.linalg.norm(y_pred_3D - y_ref, ord=2) # 开了跟号
        L2_error = np.mean(np.sqrt((y_pred_3D-y_ref)**2))
        print('Evaluation : ', z_fitness, L2_error)
        ## 按照实验时间，保存实验结果
        date_time =  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # 2022-06-18 13:49
        record_path = config['CSVPath']

        # if not os.path.exists(record_path):
        #     os.mkdir('./result/')

        EvaluateDf = pd.DataFrame({
            'Date': [date_time],
            'Epoch': [config['TrainEpoch']],
            'Intensity(mW)': [config['LightIntensity']],
            'Temperature' : [config['Temperature']],
            'Network depth' : [depth],
            'Network width' : [width],
            'Z axis fitness' : [z_fitness],
            'L2 mean error': [L2_error]
        })    

        EvaluateDf.to_csv(record_path, index=False, mode='a')

        ## 保存截面数据, X=0的YZ平面，坐标轴生成

        sample_num = 200

        def plane_YZ(x=0, sample_num = 200):
            radius_x = np.sqrt(rcell**2 -x**2)
            sample_data = []    

            for theta in np.linspace(0, 2*np.pi, sample_num):
                for r_pow in np.linspace(0, radius_x**2, sample_num):
                    y = np.sqrt(r_pow) * np.cos(theta)
                    z = np.sqrt(r_pow) * np.sin(theta)
                    sample_data.append((x, y, z)) # 200*200 = 40000个采样点

            sample_data = np.asarray(sample_data, dtype=np.float32
                                    )
            return sample_data

        YZ_axis = plane_YZ(0)
        y_pred_YZ = model.predict(YZ_axis)[:, 0]
        y_pred_YZ = np.reshape(y_pred_YZ, (sample_num, sample_num))
        YZ_df = pd.DataFrame(y_pred_YZ.T)
        YZ_file_path = os.path.join(YZ_dir_path, 'Plane_YZ_Epoch_{}.csv'.format(config['TrainEpoch']))    
        YZ_df.to_csv(YZ_file_path)   