# imports
from sklearn.linear_model import LogisticRegression
from preprocess import *
import time
from globals import *


# method to train LR
def trainAndEvaulateLR(given_seed):
  lr_start_time = time.time()
  # random split data
  X_train, X_test, y_train, y_test = splitData(given_seed)
  # print a message for debugging purposes
  print('Training LR with seed: ', given_seed)
  # train with LR
  clf = LogisticRegression(
    random_state=given_seed, 
    solver='lbfgs', 
    max_iter=2e5, 
    tol=1e-5,
    multi_class='multinomial')
  clf.fit(X_train, y_train)
  lr_end_time = time.time()
  # predict on test data
  y_pred = clf.predict(X_test)
  # evaluate the model
  cm = confusion_matrix(y_test, y_pred)
  acc = accuracy_score(y_test, y_pred)
  f = f1_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred)
  rec = recall_score(y_test, y_pred)
  auc = roc_auc_score(y_test, y_pred)
  print('Confusion Matrix: ', '\n', cm, '\n')
  return [acc, f, lr_end_time - lr_start_time, prec, rec, auc]


# execute
ans = [trainAndEvaulateLR(seed) for seed in seeds]
accs_lr = [i[0] for i in ans]
fs_lr = [i[1] for i in ans]
t_lr = [i[2] for i in ans]
prec_lr = [i[3] for i in ans]
rec_lr = [i[4] for i in ans]
auc_lr = [i[5] for i in ans]