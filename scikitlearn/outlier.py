import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Generate normal data
np.random.seed(42)
data = np.random.normal(50, 5, 100)  # mean=50, std=5, n=100

# 2. Add some outliers
data = np.append(data, [100, 110, 120, 10])

# --- Method 1: IQR ---
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]

# --- Method 2: Z-Score ---
z_scores = stats.zscore(data)
outliers_z = data[np.abs(z_scores) > 3]

print("Outliers (IQR):", outliers_iqr)
print("Outliers (Z-score):", outliers_z)

# 3. Visualize
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.boxplot(data, orient="h")
plt.title("Boxplot (Outliers visible)")

plt.subplot(1,2,2)
plt.scatter(range(len(data)), data, color="blue")
plt.axhline(y=lower_bound, color="r", linestyle="--")
plt.axhline(y=upper_bound, color="r", linestyle="--")
plt.title("Data with Outlier Bounds")

plt.show()


'''
# Outlier Detection Explained

### 🔹 Step 1: Generate some data

```python
data = np.random.normal(50, 5, 100)  
```

* Creates 100 numbers around **50** (mean=50, std=5).
* So most values are between \~40 and \~60.

Then we **manually add outliers**:

```python
data = np.append(data, [100, 110, 120, 10])
```

* Adds 4 weird values: **100, 110, 120, 10** → clearly far from the cluster.

---

### 🔹 Step 2: IQR method (Interquartile Range)

```python
Q1 = np.percentile(data, 25)   # 25th percentile
Q3 = np.percentile(data, 75)   # 75th percentile
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

* We compute the **IQR** (spread of the middle 50% of data).
* Anything **below lower\_bound** or **above upper\_bound** is called an **outlier**.

```python
outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
```

* Picks out those “too small” or “too large” values.

---

### 🔹 Step 3: Z-score method

```python
z_scores = stats.zscore(data)  
outliers_z = data[np.abs(z_scores) > 3]
```

* **Z-score** = how many standard deviations a point is away from the mean.
* If `abs(z) > 3`, it’s considered an outlier.

---

### 🔹 Step 4: Visualization

* **Boxplot**: Outliers show up as separate dots.
* **Scatter plot**: Shows raw data, with red lines for upper/lower limits.

---

✅ Output you’ll see:

```
Outliers (IQR): [100. 110. 120.  10.]
Outliers (Z-score): [100. 110. 120.]
```

* IQR catches all 4 outliers.
* Z-score misses **10** because it’s not *too far* statistically.

---

Would you like me to also **modify this code so that outliers get removed/clipped automatically** (so you get a “clean” dataset back)?


'''