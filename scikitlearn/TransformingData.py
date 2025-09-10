import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = {
    'street_name':["braodway", 'vilakazi', 'broadway', 'main'],
    'age':[25, 30, 22, 35]
}

df = pd.DataFrame(data)
print("Original DataFrame:\n", df)


# encoding categorical feature
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_encoded = onehot_encoder.fit_transform(df[['street_name']])
print("\n Encoded street names:\n", onehot_encoded)

#normalizing numerical feature
scaler = StandardScaler()
df['age_scaled'] = scaler.fit_transform(df[['age']])
print("\nDataFrame with scaled age:\n", df)

transformed_data = np.hstack((onehot_encoded, df[['age_scaled']].values))
print("\nTransformed Data (One-Hot + Scaled Age):\n", transformed_data)