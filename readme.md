# svmpy   

## Simple SVM classifier supporting linear and non linear kernels.  

### How to install.                                 
        
```pip3 install git+https://github.com/rajeshmprao/svmpy.git```     

---

### Requirements

* numpy  
* sklearn
* matplotlib   
* cvxopt

---
### Sample code. 
To run linear classifier on IRIS sample dataset with 2 features - sepal length and sepal width, with classes being iris-setosa and iris-versicolor.

``` from svmpy.utils import split_train, split_test, plot_margin, plot_contour
from svmpy.svm import SVM
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
np.place(y, y == 2, [-1])

X1 = X[50:100, :2]
X1 = X1.astype('double')

y1 = y[50:100]
y1 = y1.astype('double')

X2 = X[100:150, :2]
X2 = X2.astype('double')

y2 = y[100:150]
y2 = y2.astype('double')

X_train, y_train = split_train(X1, y1, X2, y2)
X_test, y_test = split_test(X1, y1, X2, y2)
clf = SVM(polynomial_kernel)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
correct = np.sum(y_predict == y_test)
print("%d out of %d predictions correct" % (correct, len(y_predict)))
plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

