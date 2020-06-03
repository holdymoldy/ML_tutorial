# the first two lines install the necessary packages into this notebook environment
# sklearn (scikit-learn) is a machine learning python package
# matplotlib is a plotting package meant to mimic Matlab-style plots


pip install sklearn
pip install matplotlib

from sklearn.datasets import load_iris  # downloads the iris dataset, consisting of sepal length/width
                                        # and petal length/width data of 3 different iris types (150 pts. total)
from sklearn import tree                # imports the decision tree model
import matplotlib.pyplot as plt

#  load the iris dataset
X, y = load_iris(return_X_y=True)



criterion = "entropy"  # criteria to split nodes. 'entropy' 
max_depth = 3          # maximum depth of the tree

dtree = tree.DecisionTreeClassifier(criterion=criterion, max_depth = max_depth)
dtree = dtree.fit(X, y)

plt.figure(figsize=[18,18])  # control size of the output window
output = tree.plot_tree(dtree)

# this block prints the tree in a compact way
iris = load_iris()  # re-load the data in a different format
r = tree.export_text(dtree, feature_names=iris['feature_names'])
print(r)

# if you're curious, you can play around with the max_depth or other keywords
# documentation can be found here: 
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
