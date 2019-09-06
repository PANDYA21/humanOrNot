from sklearn.linear_model import LogisticRegression
from preprocess import *

def trainAndEvaulateLR(given_seed):
  # random split data
  X_train, X_test, y_train, y_test = splitData(given_seed)
  # train with LR
  clf = LogisticRegression(
    random_state=35, 
    solver='lbfgs', 
    max_iter=2e5, 
    tol=1e-50,
    multi_class='multinomial')
  clf.fit(X_train, y_train)
  # predict on test data
  y_pred = clf.predict(X_test)
  # evaluate the model
  cm = confusion_matrix(y_test, y_pred)
  acc = accuracy_score(y_test, y_pred)
  f = f1_score(y_test, y_pred)
  print('Confusion Matrix: ')
  print(cm)
  print('\n')
  print('Accuracy: ')
  print(acc)
  print('\n')
  return [acc,f]
  # return clf,cm,acc,f


# choose some random seeds
seeds = [1,3,35,279,20]
ans = [trainAndEvaulateLR(seed) for seed in seeds]
accs_lr = [i[0] for i in ans]
fs_lr = [i[1] for i in ans]
