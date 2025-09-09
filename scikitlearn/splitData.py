import numpy as np
from sklearn.model_selection import train_test_split

x = np.random.rand(1000, 10) #features
y = np.random.randint(0, 2, 1000) #binary label(0 or 1)

# Split the data into training (70%) and temp (30%)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

# Further split the temp data into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, shuffle=True)

print("Training set shape: ", X_train.shape, y_train.shape)
print("Validation set shape: ", X_val.shape, y_val.shape)
print("Test set shape: ", X_test.shape, y_test.shape)

'''
```
What this does:

Creates a dummy dataset with 1000 rows.

Splits it into:

70% training

15% validation

15% test

Shuffling ensures random, unbiased splits.
'''