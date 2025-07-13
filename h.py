# House Price Prediction - Data Preprocessing and Feature Engineering

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew

# Step 2: Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Step 3: Initial data check
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print(train.info())

# Step 4: Handle missing values

# Drop columns with too many missing values
cols_to_drop = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)

# Fill missing categorical values with mode
cat_cols = train.select_dtypes(include='object').columns
for col in cat_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    if col in test.columns:
        test[col].fillna(test[col].mode()[0], inplace=True)

# Fill missing numerical values with median
num_cols = train.select_dtypes(exclude='object').columns
for col in num_cols:
    train[col].fillna(train[col].median(), inplace=True)
    if col in test.columns:
        test[col].fillna(test[col].median(), inplace=True)

# Step 5: Encode categorical variables

# Label encoding for ordinal features
ordinal_cols = ['ExterQual', 'BsmtQual', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual']
qual_dict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.nan: 0}
for col in ordinal_cols:
    if col in train.columns:
        train[col] = train[col].map(qual_dict)
    if col in test.columns:
        test[col] = test[col].map(qual_dict)

# One-hot encoding for nominal features
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Align the train and test set
train, test = train.align(test, join='inner', axis=1)

# Step 6: Feature engineering

# Total square footage
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

# House age
train['HouseAge'] = train['YrSold'] - train['YearBuilt']
test['HouseAge'] = test['YrSold'] - test['YearBuilt']

# Step 7: Log transformation

# Transform the target variable
train['SalePrice'] = np.log1p(train['SalePrice'])

# Transform skewed numeric features
numeric_feats = train.dtypes[train.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skewed_feats})
skewed = skewness[abs(skewness.Skew) > 0.75].index

for feat in skewed:
    train[feat] = np.log1p(train[feat])
    if feat in test.columns:
        test[feat] = np.log1p(test[feat])

# Step 8: Final shape check
print("Final Train shape:", train.shape)
print("Final Test shape:", test.shape)
