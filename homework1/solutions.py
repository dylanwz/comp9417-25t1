import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv("heart.csv")

###################################################################################################
#
#   QUESTION 1: Data Wrangling
#
###################################################################################################
y = df['Heart_Disease']

# Part (A): 
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
X_train['Age'] = X_train.apply(impute_age, axis=1)
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
k = 0.05
y_train_qnt = (y_train > k).astype(int)
y_test_qnt = (y_test > k).astype(int)
plt.clf()
plt.hist(y_train_qnt, bins=20, edgecolor='black')
plt.xlabel('Heart Disease')
plt.ylabel('Frequency')
plt.title('Target Variable Histogram')
plt.savefig('hist_cleaned.png')

X_train.to_csv('X_train_cleaned.csv', index=False)
X_test.to_csv('X_test_cleaned.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
y_train_qnt.to_csv('y_train_cleaned.csv', index=False)

# QUESTION TWO:
# B:
vals = np.logspace(-4, 4, 100)
train_losses = []
test_losses = []
for val in vals:
    m = LogisticRegression(C=val, penalty='l2', solver='lbfgs')
    m.fit(X_train, y_train_qnt) 
    y_train_prob = m.predict_proba(X_train)
    y_test_prob = m.predict_proba(X_test)
    train_losses.append(log_loss(y_train_qnt, y_train_prob))
    test_losses.append(log_loss(y_test_qnt, y_test_prob))

    y_train_pred = m.predict(X_train)
    y_test_pred = m.predict(X_test)

    train_accuracy = accuracy_score(y_train_qnt, y_train_pred)
    test_accuracy = accuracy_score(y_test_qnt, y_test_pred)

    print(f'Train Accuracy: {train_accuracy:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
plt.figure(figsize=(8, 6))
plt.plot(vals, train_losses, label='Train log loss', marker='o')
plt.plot(vals, test_losses, label='Test log loss', marker='s')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Log loss')
plt.title('Log loss vs C')
plt.legend()
plt.grid()
plt.savefig('graph.png')

# C:
# (j) Perform 5-fold cross-validation for different C values
C_values = np.logspace(-4, 4, 100)
cv_results = []

N = len(X_train)
fold_size = N // 5

for C in C_values:
    fold_losses = []
    
    for i in range(5):
        start, end = i * fold_size, (i + 1) * fold_size
        X_val, y_val = X_train.iloc[start:end], y_train_qnt.iloc[start:end]
        X_train_fold = pd.concat([X_train.iloc[:start], X_train.iloc[end:]])
        y_train_fold = pd.concat([y_train_qnt.iloc[:start], y_train_qnt.iloc[end:]])
        
        model = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=1000)
        model.fit(X_train_fold, y_train_fold)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        fold_losses.append(log_loss(y_val, y_val_prob))
    
    cv_results.append(fold_losses)

# (k) Plot boxplot of log-loss for different C values
plt.figure(figsize=(10, 6))
plt.boxplot(cv_results, positions=np.log10(C_values))
plt.xlabel('log10(C)')
plt.ylabel('Log-Loss')
plt.title('5-Fold Cross-Validation Log-Loss for Different C Values')
plt.grid()
plt.savefig('graph2')

# (l) Select best C based on median log-loss
median_losses = [np.median(losses) for losses in cv_results]
best_C = C_values[np.argmin(median_losses)]
print(f'Best C: {best_C}')

# (m) Train final model with best C
final_model = LogisticRegression(C=best_C, penalty='l2', solver='lbfgs')
final_model.fit(X_train, y_train_qnt)

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

train_accuracy = accuracy_score(y_train_qnt, y_train_pred)
test_accuracy = accuracy_score(y_test_qnt, y_test_pred)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

