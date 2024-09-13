import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def Standard_outlier(df, column1, column2): #StandardScaler method
    xy_StandardScaler = StandardScaler().fit_transform(df[[column1, column2]])
    xy_StandardScaler = np.abs(xy_StandardScaler)
    
    xy_StandardScaler_zoomin = xy_StandardScaler < 1.8
    xy_StandardScaler_zoomin = xy_StandardScaler_zoomin.astype(int)
    xy_StandardScaler_zoomin = np.multiply(xy_StandardScaler_zoomin[:,0], xy_StandardScaler_zoomin[:,1])
    
    df_res = df[xy_StandardScaler_zoomin == 1]
    df_res = df_res.reset_index(drop=True)
    
    return xy_StandardScaler_zoomin, df_res

def IQR_outlier(df, column1, column2): #IQR method
    dr = df[[column1, column2]]
    quartile_1 = dr.quantile(0.25)
    quartile_3 = dr.quantile(0.75)
    IQR = quartile_3 - quartile_1
    condition = (dr < (quartile_1 - 1.5 * IQR)) | (dr > (quartile_3 + 1.5 * IQR))
    condition = condition.any(axis=1)
    search_dr = dr[condition]
    df_res = df.drop(search_dr.index, axis=0)
    df_res = df_res.reset_index(drop=True)

    return search_dr, df_res

# example
# df1 = pd.read_excel('pos_data.xlsx', sheet_name=None, engine='openpyxl')
# search_df1, df1drop = dr_outlier(df1[str(i+1)], 'X', 'Y')