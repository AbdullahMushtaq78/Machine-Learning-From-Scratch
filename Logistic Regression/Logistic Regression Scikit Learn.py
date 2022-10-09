import numpy as np
from sklearn.linear_model import LogisticRegression
x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
pred = lr_model.predict(np.array([[0.2,1.2]]))
print("Prediction: ", pred)
print("Accuracy of the model is: ", lr_model.score(x_train, y_train)*100, "% Accurate")



