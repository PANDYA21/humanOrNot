# imports
from sklearn.neural_network import MLPClassifier
from preprocess import *
import time
from globals import *


# Method to train and evaulate MLP with tanh activation
def trainAndEvaulateMLP2(given_seed):
  mlp_start_time = time.time()
  # random split data
  X_train, X_test, y_train, y_test = splitData(given_seed)
  # print a message for debugging purposes
  print('Training MLP with seed: ', given_seed)
  # train with MLP
  clf = MLPClassifier(
    solver='lbfgs',
    activation='tanh',
    alpha=1e-4,
    max_iter=2e2,
    learning_rate='adaptive',
    verbose=False,
    hidden_layer_sizes=(800, 50),
    random_state=given_seed)
  clf.fit(X_train, y_train)
  mlp_end_time = time.time()
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
  return [acc, f, mlp_end_time - mlp_start_time, prec, rec, auc]

# execute
ans = [trainAndEvaulateMLP2(seed) for seed in seeds]
accs_mlp2 = [i[0] for i in ans]
fs_mlp2 = [i[1] for i in ans]
t_mlp2 = [i[2] for i in ans]
prec_mlp2 = [i[3] for i in ans]
rec_mlp2 = [i[4] for i in ans]
auc_mlp2 = [i[5] for i in ans]

