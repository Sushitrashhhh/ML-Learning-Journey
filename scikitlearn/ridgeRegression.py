import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(42)
x=np.linspace(0, 6,30)[: , np.newaxis]
y=np.sin(x).ravel()+np.random.normal(0, 0.3, x.shape[0])

degree =15

#no regularization
model_no_rigde = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model_no_rigde.fit(x,y)


#L2 regularization
model_rigde = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
model_rigde.fit(x,y)

#prediction

x_test = np.linspace(0, 6, 100)[:, np.newaxis]
y_no_ridge = model_no_rigde.predict(x_test)
y_ridge= model_rigde.predict(x_test)

#plotting
plt.scatter(x, y, color='black', label='Data Points')
plt.plot(x_test, y_no_ridge, color='blue', label='No Regularization', linewidth=2)
plt.plot(x_test, y_ridge, color='red', label='Ridge Regularization', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Ridge Regression vs No Regularization')
plt.legend()
plt.show()