from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

logistic_model = LogisticRegression(penalty='l2', class_weight=None, max_iter=100)
nb_model = GaussianNB()
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=5, max_leaf_nodes=500)
svm_model = SVC()

models = {"Logistic model":logistic_model, "Naive Bayes model": nb_model, "Decision Tree model" : tree_model, "SVM model" : svm_model}

for model_name, model in models.items():
    model.fit(train_data['images'], train_data['labels'])
    train_acc = model.score(train_data['images'], train_data['labels'])
    test_acc = model.score(test_data['images'], test_data['labels'])

    print(f'{model_name}: Train acc = {train_acc * 100:.2f}% ; Test acc: {test_acc * 100:.2f}%')