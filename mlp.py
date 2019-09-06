from sklearn.neural_network import MLPClassifier
from preprocess import *

def trainMLP(given_seed):
  # train with MLP
  clf = MLPClassifier(
    solver='lbfgs',
    activation='logistic',
    alpha=1e-4,
    max_iter=1e1,
    learning_rate='adaptive',
    verbose=False,
    hidden_layer_sizes=(800, 50),
    random_state=given_seed)
  clf.fit(X_train, y_train)
  return clf

def evaluate(clf):
  # predict on test data
  y_pred = clf.predict(X_test)
  # evaluate the model
  print('Confusion Matrix: ')
  print(confusion_matrix(y_test, y_pred))
  print('\n')
  print('Accuracy: ')
  print(accuracy_score(y_test, y_pred))
  print('\n')


# clf = trainMLP()
# # predict on test data
# y_pred = clf.predict(X_test)

# # evaluate the model
# print('Confusion Matrix: ')
# print(confusion_matrix(y_test, y_pred))
# print('\n')
# print('Accuracy: ')
# print(accuracy_score(y_test, y_pred))
# print('\n')

seeds = [1,3,35,279,20]
clfs = [trainMLP(seed) for seed in seeds]
[evaluate(clf) for clf in clfs]


# ans = map(trainMLP, seeds)
# # print(ans)
# print(map(evaluate, ans))
