# 生成数据集
from sklearn.datasets import make_circles

x, y = make_circles(noise=0.2, factor=0.5, random_state=1)

# 标准化
from sklearn.preprocessing import StandardScaler

x = StandardScaler().fit_transform(x)

# 可视化
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FFFFFF', '#0000FF'])
ax = plt.subplot()
ax.set_xticks(())
ax.set_yticks(())
plt.tight_layout()
plt.show()

# 产生网格点
import numpy as np

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# 使用默认的带rbf核的SVM模型进行训练
from sklearn.svm import SVC

C = 5
gamma = 0.1
clf = SVC(C=C, gamma=gamma)
clf.fit(x, y)
z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# 可视化
z = z.reshape(xx.shape)
plt.subplot()
plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.9)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.xlabel("C=" + str(C) + ',' + "gamma=" + str(gamma))
plt.show()
