#1
from sklearn.multiclass import OneVsRestClassifier

one_vs_all = OneVsRestClassifier(LogisticRegression()).fit(X, y)

plot_decision_regions(X, y, clf=one_vs_all)

from sklearn.multiclass import OneVsOneClassifier

one_vs_one = OneVsOneClassifier(LogisticRegression()).fit(X, y)

plot_decision_regions(X, y, clf=one_vs_one)