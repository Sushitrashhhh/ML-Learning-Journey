import pandas as pd
import numpy as np

data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', None],
    'age': [25, np.nan, 30, 22, np.nan, 28],
    'score' : [85, 90, np.nan, 88, 92, 95],
    'label': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df = pd.DataFrame(data)
print("Original DataFrame:", df)

print("\nHandling Missing Values: ", df.isnull().sum())

df_dropped = df.dropna()
print("\nDataFrame after dropping rows with missing values:\n", df_dropped)

df_no_dup = df.drop_duplicates()
print("\nDataFrame after removing duplicates:\n", df_no_dup)