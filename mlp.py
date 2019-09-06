from sklearn.neural_network import MLPClassifier
from preprocess import *

def trainAndEvaulateMLP(given_seed):
  # random split data
  X_train, X_test, y_train, y_test = splitData(given_seed)
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
  # predict on test data
  y_pred = clf.predict(X_test)
  # evaluate the model
  cm = confusion_matrix(y_test, y_pred)
  acc = accuracy_score(y_test, y_pred)
  f = f1_score
  print('Confusion Matrix: ')
  print(cm)
  print('\n')
  print('Accuracy: ')
  print(acc)
  print('\n')
  return acc
  # return clf,cm,acc,f


# choose some random seeds
seeds = [1,3,35,279,20]
accs_mlp = [trainAndEvaulateMLP(seed) for seed in seeds]

print(accs)
