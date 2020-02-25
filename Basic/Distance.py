# manhatun
from numpy import *
vector1 = mat([2.1,2.5,3.8])
vector2 = mat([1.0,1.7,6.6])
print(sum(abs(vector1-vector2)))

# 欧式距离
from numpy import *
vector1 = mat([2.1,2.5,3.8])
vector2 = mat([1.0,1.7,6.6])
print(sqrt((vector1-vector2)*(vector1-vector2).T))

# 切比雪夫距离
print(abs(vector1-vector2).max)

# 夹角余弦
import numpy as np
vector1 = np.array([2.1,2.5,3.8])
vector2 = np.array([1.0,1.7,6.6])
print(vector1.dot(vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)))