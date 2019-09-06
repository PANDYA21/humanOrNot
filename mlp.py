from sklearn.neural_network import MLPClassifier
from preprocess import *

def trainMLP(seed=35):
  # train with MLP
  clf = MLPClassifier(
    solver='lbfgs',
    activation='logistic',
    alpha=1e-4,
    max_iter=1e2,
    learning_rate='adaptive',
    verbose=False,
    hidden_layer_sizes=(800, 50),
    random_state=35)
  clf.fit(X_train, y_train)
  return clf


clf = trainMLP()
# predict on test data
y_pred = clf.predict(X_test)

# evaluate the model
print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_pred))
print('\n')
print('Accuracy: ')
print(accuracy_score(y_test, y_pred))
print('\n')
