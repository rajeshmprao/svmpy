from svmpy.utils import *
from svmpy.svm import SVM, gaussian_kernel, polynomial_kernel
from sklearn import datasets
def test_non_linear():
    X1, y1, X2, y2 = gen_non_lin_separable_data()
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)

    clf = SVM(gaussian_kernel)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

def test_soft():
    X1, y1, X2, y2 = gen_lin_separable_overlap_data()
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)

    clf = SVM(C=0.1)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

# test_soft()
# test_non_linear()
# print(gen_lin_separable_overlap_data())
def test_iris_linear():
    X1, y1, X2, y2 = gen_lin_separable_data()
            # X1, y1, X2, y2 = gen_non_lin_separable_data()

    print(X1.shape, y1.shape, X2.shape, y2.shape)
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    np.place(y, y == 0, [-1])

    X1 = X[0:50, :2]
    X1 = X1.astype('double')

    y1 = y[0:50]
    y1 = y1.astype('double')
    
    X2 = X[50:100, :2]
    X2 = X2.astype('double')

    y2 = y[50:100]
    y2 = y2.astype('double')
    print(X1.shape, y1.shape, X2.shape, y2.shape)
    
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)
    clf = SVM(C=0.1)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

def test_iris_nonlinear():
    X1, y1, X2, y2 = gen_lin_separable_data()
            # X1, y1, X2, y2 = gen_non_lin_separable_data()

    print(X1.shape, y1.shape, X2.shape, y2.shape)
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
test_iris_nonlinear()