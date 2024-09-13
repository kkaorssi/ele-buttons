import csv
import numpy as np
import pandas as pd
import os, math
from math import pi
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

pd.set_option('mode.chained_assignment',  None) # 경고를 끈다

df1 = pd.read_excel('./final/pos_data.xlsx', sheet_name=None, engine='openpyxl')
num_classes = 12
n_clusters=2

def dr_outlier(df, column1, column2): #StandardScaler method
    xy_StandardScaler = StandardScaler().fit_transform(df[[column1, column2]])
    xy_StandardScaler = np.abs(xy_StandardScaler)
    
    xy_StandardScaler_zoomin = xy_StandardScaler < 1.8
    xy_StandardScaler_zoomin = xy_StandardScaler_zoomin.astype(int)
    xy_StandardScaler_zoomin = np.multiply(xy_StandardScaler_zoomin[:,0], xy_StandardScaler_zoomin[:,1])
    
    df_res = df[xy_StandardScaler_zoomin == 1]
    df_res = df_res.reset_index(drop=True)
    
    return xy_StandardScaler_zoomin, df_res

def PCA(XY):
    XY_cen = XY - XY.mean(axis=0)
    XY_cov = np.dot(XY_cen.T, XY_cen) / (len(df1drop.index) - 1)
    w, v = np.linalg.eig(XY_cov)

    if (w[0] > w[1]):
        vv = v[:, 0]
    else:
        vv = v[:, 1]

    theta = np.arctan2(vv[1], vv[0])  #atan2(y,x)
    
    return theta

count_n = 0
for i in range(len(df1)):
# for i in range(10):
    if len(df1[str(i+1)]) <= 2: continue
    search_df1, df1drop = dr_outlier(df1[str(i+1)], 'X', 'Y')
    num_ins = len(df1drop.index) - len(df1drop.index[df1drop['Class'].duplicated()])
    if num_ins < num_classes / 2  or len(df1drop.index) > num_classes: continue
    
    XY = np.array(df1drop[['X', 'Y']])
    theta = PCA(XY)
    
    ab = []
    tem_V = []
    for j in range(len(df1drop)):
        a = math.tan(theta)
        b = df1drop['Y'][j] - a * df1drop['X'][j]
        ab.append([a, b])
        if j == len(df1drop)-1: continue
        xy1 = np.array([df1drop['X'][j], df1drop['Y'][j]])
        xy2 = np.array([df1drop['X'][j+1], df1drop['Y'][j+1]])
        vxy = xy1 - xy2
        dist1 = math.dist(xy1, xy2)
        tem_V.append({'Vx': vxy[0], 'Vy': vxy[1], 'dist': dist1})
        
    df2 = pd.DataFrame(tem_V)
    search_df2, df2drop = dr_outlier(df2, 'Vx', 'Vy')
        
    k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=12)
    k_means.fit(ab)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    df1drop['k_label'] = k_means_labels
    if df1drop[df1drop['k_label'] == 0]['X'].median() < df1drop[df1drop['k_label'] == 1]['X'].median():
        PF_vrf = df1drop['k_label'] == 0
    else:
        PF_vrf = df1drop['k_label'] == 1
        
    if 0.25 <= abs(theta/pi) <= 0.75:
        sort_value = 'Y'
        if df2drop['Vy'].median() > 0:
            ascending = False
        else:
            ascending = True
            
    else:
        sort_value = 'X'
        if df2drop['Vx'].median() > 0:
            ascending = False
        else:
            ascending = True
    
    if 4 <= len(df1drop[PF_vrf]) <= 6:
        tem_df1 = df1drop[PF_vrf].sort_values(by=sort_value ,ascending=ascending)
        tem_df1 = tem_df1.reset_index(drop=True)
        
        for j in range(len(tem_df1)-1):
            xy11 = [tem_df1['X'][j], tem_df1['Y'][j]]
            xy21 = [tem_df1['X'][j+1], tem_df1['Y'][j+1]]
            dist11 = math.dist(xy11, xy21)
            
            if df2drop['dist'].median()*0.8 <= dist11 <= df2drop['dist'].median()*1.2:
                continue
            elif df2drop['dist'].median()*1.75 <= dist11 <= df2drop['dist'].median()*2.25:
                new_row = pd.DataFrame([['test', (xy11[0]+xy21[0])/2, (xy11[1]+xy21[1])/2, tem_df1.loc[0, 'k_label']]], 
                                       columns = tem_df1.columns)
                tem_df1 = pd.concat([tem_df1[:j+1], new_row, tem_df1.iloc[j+1:]], ignore_index = True)
            elif df2drop['dist'].median()*2.7 <= dist11 <= df2drop['dist'].median()*3.3:
                dist_x1 = (xy21[0]-xy11[0])/3
                dist_y1 = (xy21[1]-xy11[1])/3
                new_row = pd.DataFrame([['test', xy11[0]+dist_x1, xy11[1]+dist_y1, tem_df1.loc[0, 'k_label']],
                                        ['test', xy11[0]+dist_x1*2, xy11[1]+dist_y1*2, tem_df1.loc[0, 'k_label']]], 
                                       columns = tem_df1.columns)
                tem_df1 = pd.concat([tem_df1[:j+1], new_row, tem_df1.iloc[j+1:]], ignore_index = True)
            else: continue
        
        com_res1 = []  
        for j in range(3):
            compare1 = pd.DataFrame(columns=['Class'])
            for k in range(len(tem_df1)):
                compare1.loc[k] = [j+k]
            com_res1.append(len(tem_df1['Class'].compare(compare1['Class'])))
        case1 = com_res1.index(min(com_res1))
        for j in range(6):
            if j + case1 > 6: continue
            tem_df1.loc[j, 'Class'] = j + case1
        
    
    else: continue
    
    if 4 <= len(df1drop[~PF_vrf]) <= 6:
        tem_df2 = df1drop[~PF_vrf].sort_values(by=sort_value ,ascending=ascending)
        tem_df2 = tem_df2.reset_index(drop=True)
        
        for j in range(len(tem_df2)-1):
            xy12 = [tem_df2['X'][j], tem_df2['Y'][j]]
            xy22 = [tem_df2['X'][j+1], tem_df2['Y'][j+1]]
            dist12 = math.dist(xy12, xy22)
        
            if df2drop['dist'].median()*0.8 <= dist12 <= df2drop['dist'].median()*1.2:
                continue
            elif df2drop['dist'].median()*1.75 <= dist12 <= df2drop['dist'].median()*2.25:
                new_row = pd.DataFrame([['test', (xy12[0]+xy22[0])/2, (xy12[1]+xy22[1])/2, tem_df2.loc[0, 'k_label']]], 
                                       columns = tem_df2.columns)
                tem_df2 = pd.concat([tem_df2[:j+1], new_row, tem_df2.iloc[j+1:]], ignore_index = True)
            elif df2drop['dist'].median()*2.7 <= dist12 <= df2drop['dist'].median()*3.3:
                dist_x2 = (xy22[0]-xy12[0])/3
                dist_y2 = (xy22[1]-xy12[1])/3
                new_row = pd.DataFrame([['test', xy12[0]+dist_x2, xy12[1]+dist_y2, tem_df2.loc[0, 'k_label']], 
                                        ['test', xy12[0]+dist_x2*2, xy12[1]+dist_y2*2, tem_df2.loc[0, 'k_label']]], 
                                       columns = tem_df2.columns)
                tem_df2 = pd.concat([tem_df2[:j+1], new_row, tem_df2.iloc[j+1:]], ignore_index = True)
            else: continue
 
        # for j in range(6):
        #     com_df21['Class'][j] = j + 6
    
    else: continue

    df1drop = pd.concat([tem_df1, tem_df2], ignore_index = True)
    count_n =+ 1
        
    # else:
    #     continue
    
    
    # print('theta: ', theta/pi)
    
    # if 0.495 < abs(theta/pi) < 0.505:
    #     bound = df1drop['X'].mean()
    #     PF_vrf = df1drop['X'] < bound
        
    # elif abs(theta)/pi < 0.005 or abs(theta)/pi > 0.995:
    #     bound = df1drop['Y'].mean()
    #     PF_vrf = df1drop['Y'] < bound
        
    # else:
    #     b = df1drop['Y'].mean() - a * df1drop['X'].mean()
        
    #     if theta/pi > 0.5 or -0.5 < theta/pi < 0:
    #         PF_vrf = df1drop['Y'] < a * df1drop['X'] + b
    #     else:
    #         PF_vrf = df1drop['Y'] > a * df1drop['X'] + b
            
    # if len(df1drop[PF_vrf]) == 6 and len(df1drop[~PF_vrf]) == 6:
    #     tem_df1 = df1drop[PF_vrf].sort_values(by='Y' ,ascending=False)
    #     tem_df1 = tem_df1.reset_index(drop=True)
    #     tem_df2 = df1drop[~PF_vrf].sort_values(by='Y' ,ascending=False)
    #     tem_df2 = tem_df2.reset_index(drop=True)
        
    #     for j in range(6):
    #         tem_df1['Class'][j] = j
    #         tem_df2['Class'][j] = j + 6
            
    #     df1drop = pd.concat([tem_df1, tem_df2], ignore_index = True)
    
    fig = plt.figure(2)
    
    plt.subplot(121)
    plt.title('No.%d' %i)
    plt.scatter(df1[str(i+1)]['X'], df1[str(i+1)]['Y'])
    labels = df1[str(i+1)]['Class']
    for label, XX, YY in zip(labels, df1[str(i+1)]['X'], df1[str(i+1)]['Y']):
        plt.annotate(
        label,
        xy=(XX, YY), xytext=(-10, 10),
        textcoords='offset points', ha='right', va='bottom',
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))    
        
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.subplot(122)
    
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(df1drop['k_label']))))
    for k, col in zip(range(n_clusters), colors):
        my_members = df1drop['k_label'] == k
        plt.scatter(df1drop['X'][my_members], df1drop['Y'][my_members], color = col)
    labels = df1drop['Class']
    for label, XX, YY in zip(labels, df1drop['X'], df1drop['Y']):
        plt.annotate(
        label,
        xy=(XX, YY), xytext=(-10, 10),
        textcoords='offset points', ha='right', va='bottom',
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    # if 0.495 < abs(theta/pi) and abs(theta/pi) < 0.505:
    #     plt.axvline(bound)
    #     # plt.axvline(bound, df1ymin - 100, df1ymax + 100)
        
    # elif abs(theta)/pi < 0.005 or abs(theta)/pi > 0.995:
    #     plt.axhline(bound)
        
    # else:
    x = np.array(range(int(df1drop['X'].mean()) - 50, int(df1drop['X'].mean()) + 50))
    bmean = df1drop['Y'].mean() - a * df1drop['X'].mean()
    plt.plot(x, a*x+bmean)
                
    df1xmin = df1drop['X'][df1drop['X'].idxmin()]
    df1xmax = df1drop['X'][df1drop['X'].idxmax()]
    df1ymin = df1drop['Y'][df1drop['Y'].idxmin()]
    df1ymax = df1drop['Y'][df1drop['Y'].idxmax()]

    plt.xlim([df1xmin - 100, df1xmax + 100])
    plt.ylim([df1ymin - 100, df1ymax + 100])

    plt.xlabel('X')
    
    # plt.legend()
    plt.show()