import numpy as np
import pandas as pd
import math
from outlier_removal import Standard_outlier

def PCA(df = pd.DataFrame()): # PCA
    XY = np.array(df[['X', 'Y']])
    XY_cen = XY - XY.mean(axis=0)
    XY_cov = np.dot(XY_cen.T, XY_cen) / (len(XY) - 1)
    w, v = np.linalg.eig(XY_cov)

    if (w[0] > w[1]):
        vv = v[:, 0]
    else:
        vv = v[:, 1]

    theta = np.arctan2(vv[1], vv[0]) # atan2(y,x)
    
    a = math.tan(theta)
    ab = []
    tem_V = []
    for i in range(len(df)):
        b = df['Y'][i] - a * df['X'][i]
        ab.append([a, b])
        if i == len(df)-1: continue
        xy1 = np.array([df['X'][i], df['Y'][i]])
        xy2 = np.array([df['X'][i+1], df['Y'][i+1]])
        vxy = xy1 - xy2
        dist = math.dist(xy1, xy2)
        tem_V.append({'Vx': vxy[0], 'Vy': vxy[1], 'dist': dist})
        
    dfV = pd.DataFrame(tem_V)
    search_dfV, dfVdrop = Standard_outlier(dfV, 'Vx', 'Vy')
    
    return theta, np.array(ab), dfVdrop

# example
# theta, ab, df2drop = PCA(df1drop)

def theta_median(df = pd.DataFrame()): # median
    tem_V = []
    for i in range(len(df) - 1):
        xy1 = np.array([df['X'][i], df['Y'][i]])
        xy2 = np.array([df['X'][i+1], df['Y'][i+1]])
        vxy = xy1 - xy2
        dist = math.dist(xy1, xy2)
        theta_r = np.arctan2(vxy[1], vxy[0])
        tem_V.append({'Vx': vxy[0], 'Vy': vxy[1], 'dist': dist, 'theta': theta_r})
        
    dfV = pd.DataFrame(tem_V)
    search_dfV, dfVdrop = Standard_outlier(dfV, 'Vx', 'Vy')
    
    theta_m = dfVdrop['theta'].median()
    a_median = math.tan(theta_m)
    
    ab = []
    for i in range(len(df)):
        b = df['Y'][i] - a_median * df['X'][i]
        ab.append([a_median, b])
        
    return theta_m, np.array(ab), dfVdrop
        
# example
# theta, ab, df2drop = theta_median(df1drop)