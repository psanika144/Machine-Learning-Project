import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Numerical Data
data = {
    'Age': [20, 22, 25, 30, 35],
    'Salary': [20000, 25000, 30000, 40000, 50000],
    'Department': ['IT', 'HR', 'IT', 'Sales', 'HR']
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Numerical Representation
print("\nNumerical Data:")
print(df[['Age', 'Salary']].values)

# Label Encoding
encoder = LabelEncoder()
df['Department_Code'] = encoder.fit_transform(df['Department'])

print("\nLabel Encoded Data:")
print(df)

# Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Age', 'Salary']])

print("\nStandardized Data:")
print(scaled_data)
