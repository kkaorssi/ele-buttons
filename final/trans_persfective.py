import cv2
import numpy as np

result = np.array([[500, 500],
 [580, 500],
 [580, 940],
 [500, 940]])
result = np.append(result, np.ones([4,1]), axis=1)
# mtrx = np.array([[ 3.71175205e-01, -9.37085648e-02,  3.65161634e+02],
#  [ 4.32721773e-03,  4.22509214e-01,  4.24513276e+02],
#  [ 4.22553637e-05, -1.92200415e-04,  1.00000000e+00]])

mtrx = []
org = np.array([[384.12067, 157.9997 ],
 [611.25024, 165.3576 ],
 [574.19684, 888.3108 ],
 [379.2091,  876.86945]])
org = np.append(org, np.ones([4,1]), axis=1)

for i in range(4):
    a_arr = np.delete(org, i, axis=0)
    b_arr = np.delete(result, i, axis=0)

    temp = np.zeros([3,3])
    for j in range(3):
        a = a_arr
        b = b_arr[:,j]

        temp[j,:] = np.linalg.solve(a, b)
        
        
    mtrx.append(temp)

# for i in range(4):
#     print(mtrx[i])
#     print((mtrx[i] @ org.T).T)
    
mtrx_final = (mtrx[0] + mtrx[1] + mtrx[2] + mtrx[3])/4
print(mtrx_final)
# print((mtrx_final @ org.T).T)