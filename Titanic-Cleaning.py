import pandas as pd
import numpy as np
df = pd.read_csv('/content/titanic.csv')

print("Dataset loaded successfully!")
display(df.head())
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())
# Fill missing ages using the median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked values using the mode
df['Embarked'] = df['Embarked'].fillna(
    df['Embarked'].mode()[0]
)

# Drop Cabin because it has many missing values
df = df.drop(columns=['Cabin'])

print("Missing values after cleaning:")
print(df.isnull().sum())
print("Duplicates before:", df.duplicated().sum())

df = df.drop_duplicates()
df = df.reset_index(drop=True)

print("Duplicates after:", df.duplicated().sum())
df['PassengerId'] = df['PassengerId'].astype('int64')
df['Survived'] = df['Survived'].astype('int64')
df['Pclass'] = df['Pclass'].astype('int64')
df['SibSp'] = df['SibSp'].astype('int64')
df['Parch'] = df['Parch'].astype('int64')

df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')

df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

print(df.dtypes)
df = df.rename(columns={
    'PassengerId': 'passenger_id',
    'Survived': 'survived',
    'Pclass': 'passenger_class',
    'Name': 'name',
    'Sex': 'sex',
    'Age': 'age',
    'SibSp': 'siblings_spouses',
    'Parch': 'parents_children',
    'Ticket': 'ticket',
    'Fare': 'fare',
    'Embarked': 'embarked'
})

print(df.columns.tolist())
print("Final dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

display(df.head())
df.to_csv('/content/titanic_cleaned.csv', index=False)

print("Cleaned dataset saved successfully!")
from google.colab import files

files.download('/content/titanic_cleaned.csv')
