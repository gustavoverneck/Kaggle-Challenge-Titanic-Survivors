# /src/model_dispatcher.py

from sklearn import tree, ensemble, linear_model, svm

models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    "rf": ensemble.RandomForestClassifier(),
    "log_reg": linear_model.LogisticRegression(),
    "svm": svm.SVC(C=10, gamma=0.1, kernel="rbf")
}