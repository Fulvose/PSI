from  mlxtend.plotting  import plot_decision_regions
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    print(clf.__class__.__name__)
    plt.figure(figsize=(4,4))
    plot_decision_regions(X_test,y_test,clf)
    plt.show()