import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#导入数据
PATH=r"D:\pythonProject2\pythonProject\pythonProject5\作业4\!fashion-mnist_test.csv"
DATA = numpy.loadtxt(PATH,delimiter = ",", skiprows = 1)#跳过第一行！
label = DATA[0:,0]# 加载类别标签部分

data = DATA[0:,1:]  # 加载数据部分

# x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.2,random_state = 42)
# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(data,label)
# KNeighborsClassifier()
# y_pred = model.predict(x_test)
# print(accuracy_score(y_test, y_pred))
# plot_confusion_matrix(model, x_test, y_test)
# plt.show()

from sklearn.model_selection import KFold
list=[]
dic= {}#K值对应准确率

for i in range(1,20,2):#一般k要求小于20！！！
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(data, label)
    kf = KFold(n_splits=5)
    kf.get_n_splits(data)#返回交叉迭代次数
    KFold(n_splits=5, random_state=None, shuffle=False)#不打乱
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        KNeighborsClassifier()
        y_pred = model.predict(X_test)
        ACC=accuracy_score(y_test, y_pred)
        list.append(ACC)
    array=numpy.array(list)
    ACC_mean=round(numpy.mean(array),3)
    dic[i]=ACC_mean

print("最合适的K值为：",max(dic,key=dic.get))
print(dic)


