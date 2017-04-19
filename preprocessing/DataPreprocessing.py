import pandas as pd
from io import StringIO

data_csv = """A,B,C,D
1,2,3,
5,6,7,8
9,10,,12
13,14,15,16"""
df =  pd.read_csv(StringIO(data_csv))

#Running the code below suggests there are NaN or missing values in our data
df.isnull().sum()

# The issues with missing values/NaN can be dealt with either removing the corresponding rows, columns or 
# fill them with mean,median or mode value

#1. removing rows
df.dropna()

#2. removing columns
df.dropna(axis=1)

#3. 'impute' values by filling NaNs with mean, median or most-frequent
from sklearn.preprocessing import Imputer
imt = Imputer(missing_values = 'NaN', 
              strategy = 'mean')
imt.fit(df)
X = imt.transform(df)

# HANDLING CATEGORICAL DATA
# Can be 
#  1.ordinal(can be ordered. e.g. T-shirt size)
#  2.nominal(can't be ordered. e.g. colors)

#1. Handling ordinal features
df =  pd.DataFrame([['green', 'M', 10.1, 'class1'], 
                    ['red', 'L', 13.5, 'class2'], 
                    ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'label']
# to deal with ordinal feature is to assign them a value using key-value pair dictionary with desired bigger or smaller value
dict_size = {'M': 1,
             'L': 2,
             'XL': 3}
df['size'] = df['size'].map(dict_size)
X = df.values

#2. Handling nominal features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])   # encoding color
X[:,3] = le.fit_transform(X[:,3])   # encoding class-label
X

#the nominal feature- 'color' is label-encoded as one value higher than other i.e. 0,1,2 which is undesirable since
# learning algorithm may treat them as high/small values. 
# To solve this we 'one hot encode' this feature as a sparse matrix of 0 and 1s
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()
