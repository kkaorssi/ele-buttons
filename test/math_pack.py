# from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
# import matplotlib.pyplot as plt
# from random import randint

# #new forward button callback => gets triggered, when forward button gets pushed
# def customForward(*args):
#     ax = plt.gca()
#     fig = plt.gcf()
    
#     #get line object...
#     line = ax.get_lines()[0]
    
#     #...create some new random data...
#     newData = [randint(1, 10), randint(1, 10), randint(1, 10)]
    
#     #...and update displayed data
#     line.set_ydata(newData)
#     ax.set_ylim(min(newData), max(newData))
#     #redraw canvas or new data won't be displayed
#     fig.canvas.draw()

# #monkey patch forward button callback
# NavigationToolbar2Tk.forward = customForward

# #plot first data
# plt.plot([1, 2, 3], [1, 2, 3])
# plt.show()

import math
from math import pi
import numpy as np
import matplotlib.pyplot as plt

# theta:  0.463425702378325
# direction: y=8.664769x+(-2080.153439)
# theta:  0.5488327300912401
# direction: y=-6.467154x+(2877.194489)
# theta:  0.5319697398415244
# direction: y=-9.923098x+(4580.322441)
# print(np.arctan2(0, -1)/pi)

# def test_main():
#     a = 1
#     def test():
#         nonlocal a
#         a = 2
#         print(a)
    
#     test()
#     print(a)
    
# test_main()

ab = np.array([[0, 1], [2, 3]])
print(ab[0,0])

fig = plt.figure()
sub1 = plt.subplot(121)
plt.title('before')
sub1.scatter(ab[:,0], ab[:,1])
sub2 = plt.subplot(122)
plt.title('after')
sub2.scatter(ab[:,0], ab[:,1])
plt.xlim([0,5])
plt.ylim([0,5])
plt.xlabel('X')
plt.show()
    

# org = [0, 0]
# a = [1, 0]
# b = [0, 1]
# c = np.multiply(a, [-1, -1])
# d = np.multiply(b, [-1, -1])
# e = np.array(a) + np.array(b)
# f = - np.array(a) + np.array(b)
# g = - np.array(a) - np.array(b)
# h = np.array(a) - np.array(b)
# print(e)

# theta = math.acos(np.inner(a,h)/(math.dist(org, a)*math.dist(org, h)))
# print(theta/pi)

# math.acos
# np.multiply
# np.inner,dot
# np.sum
# np.sqrt (제곱근)
# np.square
# math.dist
# math.degrees
# math.pi