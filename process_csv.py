import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

# Load the data
try:
    df = pd.read_csv('/content/data.csv',encoding='cp1252')
except:
    try:
        df = pd.read_csv('/content/data.csv',encoding='utf-8')
    except Exception as e:
        print("Please check encoding!!!")

# Print the first few rows of the DataFrame
print(df.head())
# Print the column names
print(df.columns)
# Print the data type of each column
print(df.dtypes)
# Print the number of rows and columns
print(df.shape)
# Print a summary of the DataFrame
print(df.info())
# Print the descriptive statistics of the numeric columns
print(df.describe())

# Separate the first 4 columns and the last 11 columns
first_4_columns = df.iloc[:, :4]
last_11_columns = df.iloc[:, 4:]

# Create the imputer
imputer = IterativeImputer(estimator=ExtraTreesRegressor(), max_iter=30, random_state=42)

# Fit and transform the last 11 columns
imputed_data = imputer.fit_transform(last_11_columns)

# Convert the imputed data back to a pandas DataFrame
df_imputed_11 = pd.DataFrame(imputed_data, columns=last_11_columns.columns)

# Combine the first 4 columns with the imputed last 11 columns
df_final = pd.concat([first_4_columns.reset_index(drop=True), df_imputed_11.reset_index(drop=True)], axis=1)

print(df_final)

# If you want to save the result
df_final.to_csv('/content/imputed_data.csv', index=False, encoding='utf-8')
