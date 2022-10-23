from sklearn.naive_bayes import GaussianNB
import numpy as np
#男性为1，女性为0
X = np.array([[6, 180,12], [5.92,190,11], [5.58,170,12], [5.92,165,10], [5,100,6],
              [5.5,150,8],[5.42,130,7],[5.75,150,9]])
Y = np.array([1, 1, 1, 1, 0, 0,0,0])

clf = GaussianNB()
clf.fit(X, Y)
GaussianNB()
pred=clf.predict([[6,130,8]])
print(pred)
if pred==[0]:
    print("女性")
else:
    print('男性')


