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
    if len(df1[str(i+1)]) <= 2: continue
    search_df1, df1drop = dr_outlier(df1[str(i+1)], 'X', 'Y')
    num_ins = len(df1drop.index) - len(df1drop.index[df1drop['Class'].duplicated()])
    if num_ins < num_classes-4  or len(df1drop.index) > num_classes: continue
    
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
        theta_m = np.arctan2(vxy[1], vxy[0])
        tem_V.append({'Vx': vxy[0], 'Vy': vxy[1], 'dist': dist1, 'theta_m': theta_m})
        
    df2 = pd.DataFrame(tem_V)
    search_df2, df2drop = dr_outlier(df2, 'Vx', 'Vy')
        
    k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=12)
    k_means.fit(ab)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    df1drop['k_label'] = k_means_labels
    
    theta_median = df2drop['theta_m'].median()
    a_median = math.tan(theta_median)
    b_median = df1drop['Y'].mean() - a_median * df1drop['X'].mean()
    
    abab = []
    for j in range(len(df1drop)):
        bbb = df1drop['Y'][j] - a_median * df1drop['X'][j]
        abab.append([a_median, bbb])
    
    k_median = KMeans(init="k-means++", n_clusters=n_clusters, n_init=12)
    k_median.fit(abab)
    k_median_labels = k_median.labels_
    k_median_cluster_centers = k_median.cluster_centers_
    df1drop['median_label'] = k_median_labels
    
    fig = plt.figure(figsize=(8, 8))
    
    plt.subplot(131)
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
    
    plt.subplot(132)
    
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
    
    plt.subplot(133)
    
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(df1drop['median_label']))))
    for k, col in zip(range(n_clusters), colors):
        my_members = df1drop['median_label'] == k
        plt.scatter(df1drop['X'][my_members], df1drop['Y'][my_members], color = col)
    labels = df1drop['Class']
    for label, XX, YY in zip(labels, df1drop['X'], df1drop['Y']):
        plt.annotate(
        label,
        xy=(XX, YY), xytext=(-10, 10),
        textcoords='offset points', ha='right', va='bottom',
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    plt.plot(x, a_median*x+b_median)

    plt.xlim([df1xmin - 100, df1xmax + 100])
    plt.ylim([df1ymin - 100, df1ymax + 100])

    plt.xlabel('X')

    plt.show()