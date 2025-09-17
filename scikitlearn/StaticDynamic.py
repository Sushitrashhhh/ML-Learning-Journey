import time
import random
from sklearn.linear_model import LinearRegression
import numpy as np

#train model

X=np.array([[i] for i in range(10)])
y=np.array([2*i+1 for i in range(10)])
model = LinearRegression()
model.fit(X,y)

#static inference(batch/cached)

print('====== Static inference ======')
#precompute predictions for common inputs

cache={}
for i in range(10):
    time.sleep(0.2)
    cache[i] = model.predict([[i]])[0]
print("cache built! \n")

def static_inference(x):
    return cache.get(x, "not cached!")

print("predict(3): ",static_inference(3))
print("predict(7): ",static_inference(7))
print("predict(15): ",static_inference(15))
print()

#dynamic inference
print("======= dynamic inference =======")
def dynamic_inference(x):
    time.sleep(0.2)
    return model.predict([[x]])[0]

print("predict(3): ",static_inference(3))
print("predict(7): ",static_inference(7))
print("predict(15): ",static_inference(15))