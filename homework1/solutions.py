import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv("heart.csv")

# A:
y = df['Heart_Disease']
X = df.drop(columns=['Heart_Disease', 'Last_Checkup'])

# B:
X['Age'] = X['Age'].abs()

# C:
X['Gender'] = X['Gender'].replace({'Male': 0, 'M': 0, 'Female': 1, 'F': 1, 'Unknown': 2})
X['Smoker'] = X['Smoker'].replace({'No': 0, 'N': 0, 'Yes': 1, 'Y': 1, np.nan: 2})

# D:
X[['Systolic', 'Diastolic']] = X['Blood_Pressure'].str.split('/', expand=True).astype(float)
X.drop(columns=['Blood_Pressure'], inplace=True)

# E:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# F:
medians = X_train.groupby('Gender')['Age'].median() # doesn't include NaN values
def impute_age(row):
    if pd.isnull(row['Age']):
        return medians[row['Gender']]
    return row['Age']

X_test['Age'] = X_test.apply(impute_age, axis=1)

# G: 
cols_to_noramlise = ['Age', 'Height_feet', 'Weight_kg', 'Cholesterol', 'Systolic', 'Diastolic']
def normalise_cols(df, cols):
    for col in cols:
        min_val =  df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val)/(max_val - min_val) # lwk slow but easy to read
normalise_cols(X_train, cols_to_noramlise)
normalise_cols(X_test, cols_to_noramlise)

# H:
plt.hist(y_train, bins=20, edgecolor='black')
plt.xlabel('Heart Disease')
plt.ylabel('Frequency')
plt.title('Target Variable Histogram')
plt.savefig('hist.png')
k = 0.1
y_train_cleaned = (y_train > k).astype(int)
y_test_cleaned = (y_train > k).astype(int)
plt.clf()
plt.hist(y_train_cleaned, bins=20, edgecolor='black')
plt.xlabel('Heart Disease')
plt.ylabel('Frequency')
plt.title('Target Variable Histogram')
plt.savefig('hist_cleaned.png')

X_train.to_csv('X_train_cleaned.csv', index=False)
X_test.to_csv('X_test_cleaned.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
y_train_cleaned.to_csv('y_train_cleaned.csv', index=False)

# QUESTION TWO:
vals = np.logspace(-4, 4, 100)
loss_data = []
for val in vals:
    m = LogisticRegression(C=val, solver='lbfgs').fit(X_train, y_train_cleaned)
    y_pred_train = m.predict_proba(X_train)
    y_pred_test = m.predict_proba(X_test)
    loss_data.append([val, log_loss(y_train_cleaned, y_pred_train), log_loss(y_test_cleaned, y_pred_test)])
cols = ['C value', 'train log loss', 'test log loss']
loss_df = pd.DataFrame(loss_data, columns=cols)
loss_df
