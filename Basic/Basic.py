y_true = [[0.5,1],[-1,1],[7,-6]]
y_pred = [[0,2],[-1,2],[8,-5]]

# 绝对误差（距离之差的绝对值的平均值）
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_true,y_pred)

# 均方差（差平方和的均值）
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true,y_pred)


y_true = [1,0,2,0,1,0,2,0,0,2]
y_pred = [1,0,1,0,0,0,2,0,2,1]
# 准确率
from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(y_true,y_pred,normalize=True)
print(Accuracy)

# 精度 (TP/TP+FP   查准率)
from sklearn.metrics import precision_score
Precision = precision_score(y_true,y_pred,average=None)
print(Precision)

# 召回率（TP/TP+FN  查全率）
# 输出：【类别召回率1，类别召回率2，类别召回率3】
from sklearn.metrics import  recall_score
Recall = recall_score(y_true,y_pred,average=None)
print(Recall)

# F1值 (F1 = 2*P*R / P+R)
from sklearn.metrics import f1_score
F1 = f1_score(y_true,y_pred,pos_label=1,average=None)
print(F1)

# 精度，召回率，F1值可以通过函数整体输出
from sklearn.metrics import classification_report
target_names = ['class 0','class 1','class 2']
print(classification_report(y_true,y_pred,target_names=target_names))

# ROC 曲线作用：为分类器选择最优阈值
# ROC曲线上最靠近左上角的点是分类操作最少的最优阈值
from sklearn import  metrics
print(metrics.roc_curve(y_true,y_pred,pos_label=1))

# AUC
# 当两条ROC曲线交叉时，ROC曲线下面的面积大小来比较判断，即面积大者相对更优
import numpy as np
from sklearn.metrics import roc_auc_score
roc_auc_score(np.array([0,0,1,1]),np.array([0.1,0.4,0.35,0.8]))

# 混淆矩阵
# 评估分类模型好坏的形象化展示工具
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true,y_pred,labels=[0,1,2])