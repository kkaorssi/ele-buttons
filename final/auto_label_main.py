import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from math import pi
import matplotlib.pyplot as plt
import math

# 사전 정보
NUM_CLASSES = 12 # 클래스의 개수
N_CLUSTER = 2 # 열의 개수

pd.set_option('mode.chained_assignment',  None) # 경고를 끈다

from outlier_removal import Standard_outlier
from find_theta import theta_median

df1 = pd.read_excel('../pos_data.xlsx', sheet_name=None, engine='openpyxl')
count = 0

print(len(df1))
for i in range(len(df1)):
    if len(df1[str(i+1)]) <= 2: continue
    search_df1, df1drop = Standard_outlier(df1[str(i+1)], 'X', 'Y')
    num_ins = len(set(df1drop['Class']))
    if num_ins < NUM_CLASSES-2*N_CLUSTER or len(df1drop) > NUM_CLASSES: continue
    
    XY = np.array(df1drop[['X', 'Y']])
    theta, ab, df2drop = theta_median(df1drop)
    k_means = KMeans(init="k-means++", n_clusters=N_CLUSTER, n_init=12)
    k_means.fit(ab)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    df1drop['k_label'] = k_means_labels
    
    # 메인 축의 방향
    if 0.25 <= abs(theta/pi) <= 0.75:
        sort_value = ['Y', 'X']
        if df2drop['Vy'].median() > 0:
            ascending = False
        else:
            ascending = True
            
    else:
        sort_value = ['X', 'Y']
        if df2drop['Vx'].median() > 0:
            ascending = False
        else:
            ascending = True
            
    # 서브 축의 방향(업데이트 필요)
    if (df1drop[df1drop['k_label'] == 0][sort_value[1]].median() 
        < df1drop[df1drop['k_label'] == 1][sort_value[1]].median()):
        PF_vrf = df1drop['k_label'] == 0
    else:
        PF_vrf = df1drop['k_label'] == 1

    # error correction
    if 4 <= len(df1drop[PF_vrf]) <= 6:
        tem_df1 = df1drop[PF_vrf].sort_values(by=sort_value ,ascending=ascending)
        tem_df1 = tem_df1.reset_index(drop=True)
        
        for j in range(len(tem_df1)-1):
            xy11 = [tem_df1['X'][j], tem_df1['Y'][j]]
            xy21 = [tem_df1['X'][j+1], tem_df1['Y'][j+1]]
            dist11 = math.dist(xy11, xy21)
            
            if df2drop['dist'].median()*0.8 <= dist11 <= df2drop['dist'].median()*1.2:
                continue
            elif df2drop['dist'].median()*1.6 <= dist11 <= df2drop['dist'].median()*2.4:
                new_row = pd.DataFrame([['test', (xy11[0]+xy21[0])/2, (xy11[1]+xy21[1])/2, tem_df1.loc[0, 'k_label']]], 
                                       columns = tem_df1.columns)
                tem_df1 = pd.concat([tem_df1[:j+1], new_row, tem_df1.iloc[j+1:]], ignore_index = True)
            elif df2drop['dist'].median()*2.4 <= dist11 <= df2drop['dist'].median()*3.6:
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

        if case1 > 0:
            for j in range(case1):
                dist_x = tem_df1['X'][1] - tem_df1['X'][0]
                dist_y = tem_df1['Y'][1] - tem_df1['Y'][0]
                x_pos = tem_df1['X'][0] - dist_x
                y_pos = tem_df1['Y'][0] - dist_y
                add_row = pd.DataFrame([['test', x_pos, y_pos, tem_df1.loc[0, 'k_label']]], columns = tem_df1.columns)
                tem_df1 = pd.concat([add_row, tem_df1], ignore_index = True)

        if len(tem_df1) < 6:
            for j in range(6-len(tem_df1)):
                dist_x = tem_df1['X'][len(tem_df1)-1] - tem_df1['X'][len(tem_df1)-2]
                dist_y = tem_df1['Y'][len(tem_df1)-1] - tem_df1['Y'][len(tem_df1)-2]
                x_pos = tem_df1['X'][len(tem_df1)-1] + dist_x
                y_pos = tem_df1['Y'][len(tem_df1)-1] + dist_y
                add_row = pd.DataFrame([['test', x_pos, y_pos, tem_df1.loc[0, 'k_label']]], columns = tem_df1.columns)
                tem_df1 = pd.concat([tem_df1, add_row], ignore_index = True)

        for j in range(6):
            tem_df1.loc[j, 'Class'] = j
        
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
            elif df2drop['dist'].median()*1.6 <= dist12 <= df2drop['dist'].median()*2.4:
                new_row = pd.DataFrame([['test', (xy12[0]+xy22[0])/2, (xy12[1]+xy22[1])/2, tem_df2.loc[0, 'k_label']]], 
                                       columns = tem_df2.columns)
                tem_df2 = pd.concat([tem_df2[:j+1], new_row, tem_df2.iloc[j+1:]], ignore_index = True)
            elif df2drop['dist'].median()*2.4 <= dist12 <= df2drop['dist'].median()*3.6:
                dist_x2 = (xy22[0]-xy12[0])/3
                dist_y2 = (xy22[1]-xy12[1])/3
                new_row = pd.DataFrame([['test', xy12[0]+dist_x2, xy12[1]+dist_y2, tem_df2.loc[0, 'k_label']], 
                                        ['test', xy12[0]+dist_x2*2, xy12[1]+dist_y2*2, tem_df2.loc[0, 'k_label']]], 
                                       columns = tem_df2.columns)
                tem_df2 = pd.concat([tem_df2[:j+1], new_row, tem_df2.iloc[j+1:]], ignore_index = True)
            else: continue
        
        com_res2 = []  
        for j in range(3):
            compare2 = pd.DataFrame(columns=['Class'])
            for k in range(len(tem_df2)):
                compare2.loc[k] = [j+k+6]
            com_res2.append(len(tem_df2['Class'].compare(compare2['Class'])))
        case2 = com_res2.index(min(com_res2))

        if case2 > 0:
            for j in range(case2):
                dist_x = tem_df2['X'][1] - tem_df2['X'][0]
                dist_y = tem_df2['Y'][1] - tem_df2['Y'][0]
                x_pos = tem_df2['X'][0] - dist_x
                y_pos = tem_df2['Y'][0] - dist_y
                add_row = pd.DataFrame([['test', x_pos, y_pos, tem_df2.loc[0, 'k_label']]], columns = tem_df2.columns)
                tem_df2 = pd.concat([add_row, tem_df2], ignore_index = True)

        if len(tem_df2) < 6:
            for j in range(6-len(tem_df2)):
                dist_x = tem_df2['X'][len(tem_df2)-1] - tem_df2['X'][len(tem_df2)-2]
                dist_y = tem_df2['Y'][len(tem_df2)-1] - tem_df2['Y'][len(tem_df2)-2]
                x_pos = tem_df2['X'][len(tem_df2)-1] + dist_x
                y_pos = tem_df2['Y'][len(tem_df2)-1] + dist_y
                add_row = pd.DataFrame([['test', x_pos, y_pos, tem_df2.loc[0, 'k_label']]], columns = tem_df2.columns)
                tem_df2 = pd.concat([tem_df2, add_row], ignore_index = True)
 
        for j in range(6):
            tem_df2.loc[j, 'Class'] = j + 6
    
    else: continue

    df1drop = pd.concat([tem_df1, tem_df2], ignore_index = True)
    count_n =+ 1
        
    ## visualize 

    fig = plt.figure(i, figsize=(10, 5))

    plt.subplot(121)
    plt.title('before')

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
    plt.title('after')
    
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(df1drop['k_label']))))
    for k, col in zip(range(len(df1drop)), colors):
        my_members = df1drop['k_label'] == k
        plt.scatter(df1drop['X'][my_members], df1drop['Y'][my_members], color = col)
    labels = df1drop['Class']
    for label, XX, YY in zip(labels, df1drop['X'], df1drop['Y']):
        plt.annotate(
        label,
        xy=(XX, YY), xytext=(-10, 10),
        textcoords='offset points', ha='right', va='bottom',
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        
    x = np.array(range(int(df1drop['X'].mean()) - 50, int(df1drop['X'].mean()) + 50))
    a = ab[0,0]
    b = np.median(ab[:,1])
    plt.plot(x, a*x+b)
                
    df1xmin = df1drop['X'][df1drop['X'].idxmin()]
    df1xmax = df1drop['X'][df1drop['X'].idxmax()]
    df1ymin = df1drop['Y'][df1drop['Y'].idxmin()]
    df1ymax = df1drop['Y'][df1drop['Y'].idxmax()]

    plt.xlim([df1xmin - 100, df1xmax + 100])
    plt.ylim([df1ymin - 100, df1ymax + 100])

    plt.xlabel('X')
    
    plt.show()