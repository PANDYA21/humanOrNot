from sklearn.linear_model import LogisticRegression
from preprocess import *

# train with LR
clf = LogisticRegression(random_state=35, solver='lbfgs', max_iter=2e5, tol=1e-50,
	multi_class='multinomial').fit(X_train, y_train)

# predict on test data
y_pred = clf.predict(X_test)

# evaluate the model
print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_pred))
print('\n')
print('Accuracy: ')
print(accuracy_score(y_test, y_pred))
print('\n')
