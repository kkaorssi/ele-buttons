import pandas as pd
import numpy as np
import csv

# df = pd.DataFrame(np.random.randn(10,4),
#                   columns=['A', 'B', 'C', 'D'])

# df.to_csv('result.csv')

data = pd.read_csv('./result.csv')
print(data)