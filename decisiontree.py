from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

#assign petal length and width to x, classification options to y
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
print 'Class labels:', np.unique(y)

#split data into test data and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print np.bincount(y), np.bincount(y_train), np.bincount(y_test)

#feature scaling for optimal performance
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#inintialize decision tree
tree = DecisionTreeClassifier(criterion='gini', max_depth = 4, random_state=1)
tree.fit(X_train, y_train)

#visualized results
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')

    if test_idx:
        X_test, y_test = X[test_idx,:], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1], c='', edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=100, label='test set')

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105,150))
plt.xlabel('standardized petal length')
plt.ylabel('standardized petal width')
plt.legend(loc='upper left')
plt.show()

dot_data = export_graphviz(tree, filled=True, rounded=True, class_names=['Setosa','Versicolor','Virginica'], feature_names=['petal length','petal width'], out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png') #brew install graphviz
