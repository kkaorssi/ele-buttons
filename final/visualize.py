from re import I
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualization(i, df1, df1drop, ab):
    
    fig = plt.figure(i, figsize=(10, 5))

    plt = plt.subplot(121)
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