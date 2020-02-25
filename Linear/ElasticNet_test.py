# 弹性网络回归

# 导入数据fsads
from sklearn.datasets import load_boston
boston = load_boston()
x = boston.data
y = boston.target
print(x.shape)
print(y.shape)

# 划分数据集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)

# 数据标准化
from sklearn import preprocessing
standard_x = preprocessing.StandardScaler()
X_train = standard_x.fit_transform(X_train)
X_test = standard_x.transform(X_test)

standard_y = preprocessing.StandardScaler()
y_train = standard_y.fit_transform(y_train.reshape(-1, 1))
y_test = standard_y.transform(y_test.reshape(-1, 1))

# 运用ElasticNet回归模型训练和预测
from sklearn.linear_model import ElasticNet
ElasticNet_clf = ElasticNet(alpha=0.1, l1_ratio=0.71)
ElasticNet_clf.fit(X_train, y_train.ravel())
ElasticNet_clf_socre = ElasticNet_clf.score(X_test, y_test.ravel())
print('lasso模型得分', ElasticNet_clf_socre)
print('特征权重', ElasticNet_clf.coef_)
print('偏置值', ElasticNet_clf.intercept_)
print('迭代次数', ElasticNet_clf.n_iter_)

# 画图
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,3))
axes = fig.add_subplot(1,1,1)
line1, = axes.plot(range(len(y_test)), y_test, 'b', label='Actual Value')
ElasticNet_clf_result = ElasticNet_clf.predict(X_test)
line2, = axes.plot(range(len(ElasticNet_clf_result)),ElasticNet_clf_result,'r--', label='ElasticNet_Predicted', linewidth=2)
axes.grid()
fig.tight_layout()
plt.legend(handles=[line1, line2])
plt.title('ElasticNet')
plt.show()