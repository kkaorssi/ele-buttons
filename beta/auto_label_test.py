import csv
import numpy as np
import pandas as pd
import os, math
from math import pi
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import openpyxl

pd.set_option('mode.chained_assignment',  None) # 경고를 끈다

df1 = pd.read_excel('../pos_data.xlsx', sheet_name=None, engine='openpyxl')
num_classes = 12

def dr_outlier(df, column1, column2): #StandardScaler method
    xy_StandardScaler = StandardScaler().fit_transform(df[[column1, column2]])
    xy_StandardScaler = np.abs(xy_StandardScaler)
    
    xy_StandardScaler_zoomin = xy_StandardScaler < 1.8
    xy_StandardScaler_zoomin = xy_StandardScaler_zoomin.astype(int)
    xy_StandardScaler_zoomin = np.multiply(xy_StandardScaler_zoomin[:,0], xy_StandardScaler_zoomin[:,1])
    
    df_res = df[xy_StandardScaler_zoomin == 1]
    
    return xy_StandardScaler_zoomin, df_res

count_n = 0
A = [1, 0]

for i in range(len(df1)):
# for i in range(10):
    if len(df1[str(i+1)]) <= 2: continue
    search_df1, df1drop = dr_outlier(df1[str(i+1)], 'X', 'Y')
    num_ins = len(df1drop.index) - len(df1drop.index[df1drop['Class'].duplicated()])

    tem_V = []
    # if num_ins >= num_classes / 2  and len(df1drop.index) <= num_classes:
    # if len(df1drop.index) == num_classes:
    if num_ins >= num_classes / 2:
    
        for j in range(len(df1drop.index)-1):
            xy1 = np.array([df1drop['X'][df1drop.index[j]], df1drop['Y'][df1drop.index[j]]])
            xy2 = np.array([df1drop['X'][df1drop.index[j+1]], df1drop['Y'][df1drop.index[j+1]]])
            vxy = xy1 - xy2
            # theta = math.acos(np.inner(A, vxy)/(math.dist(xy1, xy2)))/pi
            theta = np.arctan2(vxy[1], vxy[0])
            tem_V.append({'Vx': vxy[0], 'Vy': vxy[1], 'theta': theta})
            
        df2 = pd.DataFrame(tem_V)
        search_df2, df2drop = dr_outlier(df2, 'Vx', 'Vy')
        
        # y 값의 역순이라고 가정 => 버튼의 순서 찾는 코드 필요
        df1drop = df1drop.sort_values(by='Y' ,ascending=False)
        df1drop = df1drop.reset_index(drop=True)
        
        # 버튼의 배치 방향 찾기
        in_angle = df2drop['theta'].mean()/pi
        print('theta: ', in_angle)
        
        if 0.495 < in_angle and in_angle < 0.505:
            bound = df1drop['X'].mean()
            PF_vrf = df1drop['X'] < bound
            print('direction: x=%f' %(bound))
            
        elif in_angle < 0.005 or in_angle > 0.995:
            bound = df1drop['Y'].mean()
            PF_vrf = df1drop['Y'] < bound
            print('direction: y=%f' %(bound))
            
        else:
            a = math.tan(in_angle*pi)
            b = df1drop['Y'].mean() - a * df1drop['X'].mean()
            print('direction1: y=%fx+(%f)' %(a, b))
            print('mean: ', df1drop['X'].mean(), df1drop['Y'].mean())
            
            if in_angle > 0.5:
                PF_vrf = df1drop['Y'] < a * df1drop['X'] + b
            else:
                PF_vrf = df1drop['Y'] > a * df1drop['X'] + b

        x = np.array(range(int(df1drop['X'].mean()) - 50, int(df1drop['X'].mean()) + 50))
            
        if len(df1drop[PF_vrf]) == 6 and len(df1drop[~PF_vrf]) == 6:
            tem_df1 = df1drop[PF_vrf].sort_values(by='Y' ,ascending=False)
            tem_df1 = tem_df1.reset_index(drop=True)
            tem_df2 = df1drop[~PF_vrf].sort_values(by='Y' ,ascending=False)
            tem_df2 = tem_df2.reset_index(drop=True)
            
            # if len(df1drop[PF_vrf]) == 6:
            for j in range(6):
                tem_df1['Class'][j] = j
                    
            # if len(df1drop[~PF_vrf]) == 6:
            # for j in range(len(tem_df2)):
                tem_df2['Class'][j] = j + 6
                    
            df1drop = pd.concat([tem_df1, tem_df2], ignore_index = True)
            # df1drop[~PF_vrf].sort_values(by='Y' ,ascending=False)
            
            count_n += 1
            
# print(count_n)
            # print('after: ', df1drop)

        ## 시각화
        
        df1xmin = df1drop['X'][df1drop['X'].idxmin()]
        df1xmax = df1drop['X'][df1drop['X'].idxmax()]
        df1ymin = df1drop['Y'][df1drop['Y'].idxmin()]
        df1ymax = df1drop['Y'][df1drop['Y'].idxmax()]
        
        df2xmin = df2drop['Vx'][df2drop['Vx'].idxmin()]
        df2xmax = df2drop['Vx'][df2drop['Vx'].idxmax()]
        df2ymin = df2drop['Vy'][df2drop['Vy'].idxmin()]
        df2ymax = df2drop['Vy'][df2drop['Vy'].idxmax()]
        
        fig = plt.figure()
        fig.set_facecolor('white')

        font_size = 15
        labels = df1drop['Class']
        
        # plt.subplot(121)
        
        plt.scatter(df1drop['X'],df1drop['Y']) ## 원 데이터 산포도
        # plt.text(df1xmin - 100, df1ymax + 120, 'df1: ' + str(len(df1drop.index)), fontsize=12)
        # plt.text(df1xmin + 50, df1ymax + 120, 'df2: ' + str(len(df2drop.index)), fontsize=12)
        
        # if len(df1drop[PF_vrf]) == 6 and len(df1drop[~PF_vrf]) == 6:
        #     plt.title('Pass')
        # else:
        #     plt.title('Fail')
        plt.title('No.%d' %i)
        
        for label, XX, YY in zip(labels, df1drop['X'], df1drop['Y']):
            plt.annotate(
                label,
                xy=(XX, YY), xytext=(-10, 10),
                textcoords='offset points', ha='right', va='bottom',
                # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        
        if 0.495 < in_angle and in_angle < 0.505:
            plt.axvline(bound)
            # plt.axvline(bound, df1ymin - 100, df1ymax + 100)
            
        elif in_angle < 0.005 or in_angle > 0.995:
            plt.axhline(bound)
            
        else:
            plt.plot(x, a*x+b)
        
        plt.xlim([df1xmin - 100, df1xmax + 100])
        plt.ylim([df1ymin - 100, df1ymax + 100])
    
        plt.xlabel('X', fontsize=font_size)
        plt.ylabel('Y',fontsize=font_size)
        
        # plt.subplot(122)
        # plt.scatter(df2drop['Vx'],df2drop['Vy'])
        
        # plt.xlim([df2xmin - 10, df2xmax + 10])
        # plt.ylim([df2ymin - 10, 0])
        
        # plt.xlabel('X', fontsize=font_size)
        
        plt.show()