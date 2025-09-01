import pandas as pd 

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red', 'Green']})

print("Original Data:  ", data)

#vocabulary encoding
label_encoder = LabelEncoder()
data['Color_Label'] = label_encoder.fit_transform(data['Color'])
print("Label Encoded(indexing):  ", data)


#one hot encoding
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_encoded = onehot_encoder.fit_transform(data[['Color']])
onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['Color']))

final_data = pd.concat([data, onehot_df], axis=1)
print("One Hot Encoded:  ", final_data)

'''
```
$ python onehot.py

Original Data:
   Color
0    Red
1   Blue
2  Green
3   Blue
4    Red
5  Green

--- Label Encoding (unique index assign) ---
Unique Categories (sorted): ['Blue', 'Green', 'Red']
Encoded Mapping:
Blue  -> 0
Green -> 1
Red   -> 2

After Label Encoding:
   Color  Color_Label
0    Red            2
1   Blue            0
2  Green            1
3   Blue            0
4    Red            2
5  Green            1

--- One Hot Encoding (binary columns) ---
Unique Categories: ['Blue', 'Green', 'Red']

One Hot Encoded Matrix:
   Color_Blue  Color_Green  Color_Red
0         0.0          0.0        1.0
1         1.0          0.0        0.0
2         0.0          1.0        0.0
3         1.0          0.0        0.0
4         0.0          0.0        1.0
5         0.0          1.0        0.0

--- Final Data (concatenated) ---
   Color  Color_Label  Color_Blue  Color_Green  Color_Red
0    Red            2         0.0          0.0        1.0
1   Blue            0         1.0          0.0        0.0
2  Green            1         0.0          1.0        0.0
3   Blue            0         1.0          0.0        0.0
4    Red            2         0.0          0.0        1.0
5  Green            1         0.0          1.0        0.0

```
'''