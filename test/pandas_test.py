import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10,3),columns=['A','B','C'])

print('1: ', df)

# df.loc[10, 'A'] = 10
# print(df.loc[0, 'A'])

# print('2: ', df['A'].median())

df1 = df.copy()

df1.loc[0, 'A'] = 1000
df1.loc[3, 'A'] = 1000
df1.loc[10, 'A'] = 1000

print('2: ', df1)

res = df['A'].compare(df1['A'])

# print('2: ', df1)

# print('compared: ', res, type(res), len(res))
# print(df)
# print('3: ', df['A'].loc[df['A'].idxmax()])